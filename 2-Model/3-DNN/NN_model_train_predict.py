'''

This file is a modified version of batch_policy_learning_nn.py

Instead of training a single neural net, we will train 25 neural nets, each corresponding to a type of coupon

TRAINING
----------

1. Split training data into 25 groups by action_type
2. For each group, train a neural net and save it
    a) Features: 375 state variables 
    b) Target: reward
3. We will end up creating 25 models

EVALUATION
----------
1. For each row of unseen data
    a) Predict reward on each of the 25 trained models
    b) Output the action associated with the model that yields the highest predicted reward

Required package: PyTorch v1.2.0

'''
from __future__ import division

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import os
from time import time
import datetime
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import sys
sys.path.append('..')
from input_file import input_file

class BaselineNNInstance25Models(object):

    def __init__(self, num_state_var, num_action, dropout_rate=0.2):
        '''
        Parameters
        ----------
        num_state_var: int
            Number of state variables
        num_action: int
            Number of action types
        dropout_rate: float, default 0.2
            Dropout rate used in FC layers

        '''
        # input_dim = num_state_var + num_action
        self.model_list = [BaselineFCStates(num_state_var, dropout_rate) for _ in range(num_action)]
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        # self.model.to(self.device)
        self.num_state_var = num_state_var
        self.num_action = num_action

    def loadWeights(self, model_path):
        '''
        Load saved weights into model
        '''
        for i, m in enumerate(self.model_list):
            m.load_state_dict(torch.load('{}/nn_action_{}.pt'.format(model_path, i)))
        print('Weights loaded from {}'.format(model_path))

    def saveWeights(self, model_path):
        '''
        Save weights
        '''
        for i, m in enumerate(self.model_list):
            torch.save(m.state_dict(), '{}/nn_action_{}.pt'.format(model_path, i))
        print('Weights saved to {}'.format(model_path))

    def trainAll(self,
                train_data,
                states_val=None,
                rewards_val=None,
                num_epochs=10,
                batch_size=32,
                lr=0.005,
                eval_on_valid=False):
        '''
        Train all models in self.model_list
        Takes in all train data, group by action_type, train separately

        '''
        
        # Split training data into 25 actions
        action_matrix = train_data.iloc[:, 2 + self.num_state_var: 2 + self.num_state_var + self.num_action].values
        print('action_matrix has shape {}'.format(action_matrix.shape))
        train_data['action'] = action_matrix.argmax(axis=1)
        
        train_grp = train_data.groupby('action')

        for action_type, df in train_grp:
            print('*' * 60)
            print('*' * 20, 'TRAINING MODEL FOR ACTION_TYPE {}'.format(action_type), '*' * 20)
            print('*' * 60)
            print()

            df = df.drop(columns=['action'])
            print('df has shape {}'.format(df.shape))

            # Get state and reward vars
            states_train, rewards_train = df.iloc[:, 2: 2 + self.num_state_var].values, df.iloc[:, -1].values

            # Train a model
            self.train(action_type,
                    states_train,
                    rewards_train,
                    states_val,
                    rewards_val,
                    num_epochs,
                    batch_size,
                    lr,
                    eval_on_valid)


    def train(self,
            action_type,
            states_train,
            rewards_train,
            states_val=None,
            rewards_val=None,
            num_epochs=10,
            batch_size=32,
            lr=0.005,
            eval_on_valid=False):
        '''
        Train a single model in self.model_list using training data associated with the specified action_type
        RETURN model
        
        '''

        # Turn training data into DataLoaders
        dataset_train = TensorDataset(torch.Tensor(states_train),
                                    torch.Tensor(rewards_train).unsqueeze(dim=1)) # Convert rewards_train from 1-d to 2-d

        dataloader_train = DataLoader(dataset_train, batch_size=batch_size)

        # Turn validation data into Dataloaders
        if eval_on_valid:
            dataset_val = TensorDataset(torch.Tensor(states_val),
                                    torch.Tensor(rewards_val).unsqueeze()) # Convert rewards_train from 1-d to 2-d

            dataloader_val = DataLoader(dataset_val, batch_size=batch_size)


        # Let model reference self.model 
        model = self.model_list[action_type]

        # Criterion
        criterion = nn.MSELoss()
        # Optimizer
        optimizer = optim.Adam([param for param in model.parameters() if param.requires_grad], lr=lr)

        optimizer.zero_grad()

        for epoch in range(num_epochs):
            # Train
            model.train()
            print('=' * 30, 'Epoch {}'.format(epoch + 1), '=' * 30)
            train_loss = 0
            for i, (s, t) in enumerate(dataloader_train):
                optimizer.zero_grad()
                output = model(s)
                loss = criterion(output, t)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
                
                # print average training loss every 50 steps
                if i % 50 == 0 or i == len(dataloader_train) - 1:
                    print('Step {}/{}, Avg train loss {}'.format(i+1, len(dataloader_train), train_loss / (i+1)))
            
            # Evaluate on valid set
            if eval_on_valid:
                with torch.no_grad():
                    model.eval()
                    val_loss = 0
                    for s, t in dataloader_val:
                        output = model(s)
                        loss = criterion(output, t)
                        val_loss += loss.item()
                    
                    # print average validation loss
                    print('Avg val loss {}'.format(val_loss / len(dataloader_val)))

        return model


    def predictOptimalAction(self, states, time_ids):
        '''
        Predict the action that the model believes will induce the maximum reward
        
        '''
#         if not isinstance(states, np.ndarray):
#             raise TypeError('Expected input to be a numpy array but got {}'.format(type(states)))
        
#         print('Predicting...')
        assert states.size(1) == self.num_state_var

        # Convert state vars into Tensors
#         states = torch.Tensor(states)

        # Initialize an array to record predicted rewards
        resArray = np.zeros((states.shape[0], self.num_action))

        # For each of the 25 models, predict rewards and save them in resArray
        for action_type in range(self.num_action):
            
#             print('At action_type {}'.format(action_type))

            model = self.model_list[action_type]
            model.eval()
        
            with torch.no_grad():
                pred_rewards = model(states)
                resArray[:, action_type] = pred_rewards.detach().numpy().ravel()
        
        # Reformat result
        res = resArray.argmax(axis=1)
#         res = np.eye(self.num_action)[resArrayArgMax]
        time_ids = time_ids.squeeze().numpy()
    
        return res, time_ids


################## Underlying Neural Net ##################

class BaselineFCStates(nn.Module):

    def __init__(self, input_dim, dropout_rate=0.2):
        super(BaselineFCStates, self).__init__()
        # Network architecture
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, states):
        out = self.dropout(self.fc1(states))
        out = F.relu(out)
        out = self.dropout(self.fc2(out))
        out = F.relu(out)
        out = self.fc3(out)
        return out


if __name__ == "__main__":
    num_state_var = 375
    num_action = 25
    t0 = time()

    # make folder to save saved model
    today = datetime.date.today().strftime('%y%m%d')
    folder = os.path.join('../', 'saved_model_' + str(today))
    folder = os.path.join(folder , 'nn_25_models')
    if not os.path.exists(folder):
        os.makedirs(folder)

    model_nn = BaselineNNInstance25Models(num_state_var, num_action, dropout_rate=0.2)
    ''' Train '''
    # plug in data
    # read_filename = '../../../smalldata/model_input_sample_small_train_log.csv'
    read_filename = input_file
    train_data = pd.read_csv(read_filename)
    print('Input data shape:', train_data.shape)


    model_nn.trainAll(train_data,
                num_epochs=10,
                batch_size=32,
                lr=0.005)
    model_nn.saveWeights(folder)
