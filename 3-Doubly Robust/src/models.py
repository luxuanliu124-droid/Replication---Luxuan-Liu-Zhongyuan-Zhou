import torch.nn as nn
import torch
import copy
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np
from src.config import live_config as config
from sklearn.ensemble import RandomForestClassifier

# Gaomin 20191228: Add done classification model to predict done value of t+1 given s_t and a_t

class DonePredict():
    def __init__(self, n_estimators=50, max_depth=5):
        self.model  = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

    def train(self, traj_set):
        # read data from traj_set
        num_samples = len(traj_set)
        # num_samples = config.num_samples
        traj_len = np.zeros(num_samples, 'int')
        state_arr = np.zeros((num_samples, config.max_length, config.state_dim))
        action_arr = np.zeros((num_samples, config.max_length, 1))
        done_arr = np.ones((num_samples, config.max_length))

        for i_traj in range(num_samples):
            traj_len[i_traj] = len(traj_set.trajectories[i_traj])
            state_arr[i_traj, 0:traj_len[i_traj], :] = np.concatenate([t.state for t in traj_set.trajectories[i_traj]])
            action_arr[i_traj, 0:traj_len[i_traj], :] = np.concatenate([t.action for t in traj_set.trajectories[i_traj]])
            done_arr[i_traj, 0:traj_len[i_traj]] = 0
            if traj_len[i_traj] < config.max_length:
                done_arr[i_traj, traj_len[i_traj]:] = 1

        done_3d = done_arr[:, :, np.newaxis]
        done_next = np.roll(done_arr, -1)
        done_next[:, -1] = 1
        done_next_3d = done_next[:, :, np.newaxis]
        done_data = np.concatenate([state_arr, action_arr, done_3d, done_next_3d], axis=-1)
        done_data_2d = done_data.reshape(-1, done_data.shape[-1])
        # keep only data of done_t = 0
        done_data_2d = done_data_2d[np.where(1 - done_data_2d[:, -2])]
        # drop column of done_t
        done_data_2d = np.delete(done_data_2d, -2, axis=1)
        X = done_data_2d[:, :-1]
        Y = done_data_2d[:, -1]
        self.model.fit(X, Y)
        print('-' * 20)
        print('- done model trained -')
        print('-' * 20)


    def forward(self, state, action):
        state_arr = state.numpy()
        action_arr = action.numpy()
        action_arr = action_arr.reshape(-1, 1)
        X = np.concatenate([state_arr, action_arr], axis=1)
        done_pred = self.model.predict(X)

        return done_pred

class QNet(nn.Module):
    def __init__(self, state_dim, hidden_dims, action_size):
        super(QNet, self).__init__()
        mlp_layers = []
        prev_hidden_size = state_dim
        for next_hidden_size in hidden_dims:
            mlp_layers.extend([
                nn.Linear(prev_hidden_size, next_hidden_size),
                nn.Tanh(),
            ])
            prev_hidden_size = next_hidden_size
        mlp_layers.append(
            nn.Linear(prev_hidden_size, action_size)
        )
        self.model = nn.Sequential(*mlp_layers)

    def forward(self, state):
        return self.model(state)


class QtNet(nn.Module):
    def __init__(self, state_dim, hidden_dims, action_size):
        super(QtNet, self).__init__()
        self.action_size = action_size
        self.state_dim = state_dim
        mlp_layers = []
        prev_hidden_size = state_dim
        for next_hidden_size in hidden_dims:
            mlp_layers.extend([
                nn.Linear(prev_hidden_size, next_hidden_size),
                nn.Tanh(),
            ])
            prev_hidden_size = next_hidden_size
        mlp_layers.append(
            nn.Linear(prev_hidden_size, action_size)
        )
        self.model = nn.Sequential(*mlp_layers)
        self.time_weights = nn.Linear(1, 1)

    def forward(self, state, time):
        #f = torch.cat((state,time),1)
        #return self.model(f)
        return self.model(state)+self.time_weights(time).repeat(1,self.action_size)


class MDPnet(nn.Module):
    def __init__(self, config):
        super(MDPnet, self).__init__()
        self.config = config
        # representation
        mlp_layers = []
        prev_hidden_size = config.state_dim
        for next_hidden_size in config.rep_hidden_dims:
            mlp_layers.extend([
                nn.Linear(prev_hidden_size, next_hidden_size),
                nn.Tanh(),
            ])
            prev_hidden_size = next_hidden_size
        self.representation = nn.Sequential(*mlp_layers)
        self.rep_dim = prev_hidden_size

        # Transition
        mlp_layers = []
        for next_hidden_size in config.transition_hidden_dims:
            mlp_layers.extend([
                nn.Linear(prev_hidden_size, next_hidden_size),
                nn.Tanh(),
            ])
            prev_hidden_size = next_hidden_size
        mlp_layers.append(
            nn.Linear(prev_hidden_size,config.action_size*config.state_dim)
        )
        self.transition = nn.Sequential(*mlp_layers)

        #Reward
        mlp_layers = []
        prev_hidden_size = self.rep_dim
        for next_hidden_size in config.reward_hidden_dims:
            mlp_layers.extend([
                nn.Linear(prev_hidden_size, next_hidden_size),
                nn.Tanh(),
            ])
            prev_hidden_size = next_hidden_size
        mlp_layers.append(
            nn.Linear(prev_hidden_size, config.action_size)
        )
        self.reward = nn.Sequential(*mlp_layers)

    def forward(self,state):
        rep = self.representation(state)
        next_state_diff = self.transition(rep).view(-1,self.config.action_size,self.config.state_dim)
        reward = self.reward(rep).view(-1,self.config.action_size)

        # Gaomin: change_padding add padding to reward 12/30/19
        pad = torch.zeros(reward.size()[0],1)
        reward = torch.cat((reward, pad), axis=1)
        pad = torch.zeros(next_state_diff.size()[0], 1, next_state_diff.size()[2])
        next_state_diff = torch.cat((next_state_diff, pad), axis=1)

        #soft_done = self.terminal(state)
        return next_state_diff, reward, rep

    # oracle for cartpole, we should not use
    def get_isdone(self,state):
        x = state[0,0]
        theta = state[0,2]
        x_threshold = 2.4
        theta_threshold_radians = 12*2*math.pi/360
        done = x < -x_threshold \
               or x > x_threshold \
               or theta < -theta_threshold_radians \
               or theta > theta_threshold_radians
        done = bool(done)
        return done
    # oracle for cartpole, we should not use
    def get_reward(self,state):
        return self.config.oracle_reward


class TerminalClassifier(nn.Module):
    def __init__(self, config):
        super(TerminalClassifier, self).__init__()
        self.config = config

        mlp_layers = []
        prev_hidden_size = self.config.state_dim #self.config.rep_hidden_dims[-1]
        for next_hidden_size in config.terminal_hidden_dims:
            mlp_layers.extend([
                nn.Linear(prev_hidden_size, next_hidden_size),
                nn.Tanh(),
            ])
            prev_hidden_size = next_hidden_size
        mlp_layers.extend([
            nn.Linear(prev_hidden_size, 1),
        ])
        self.terminal = nn.Sequential(*mlp_layers)

    def forward(self, x):
        return self.terminal(x)


