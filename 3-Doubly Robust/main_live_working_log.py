import argparse
import pickle
import torch.optim as optim
import torch
from time import time
from sklearn.externals import joblib
from collections import deque
from src.memory import *
from src.utils import *
from src.models import MDPnet, DonePredict
from src.config import live_config
from live_domain.batch_policy_learning import BatchPolicyLearning
from live_domain.live_simulator.batch_data_processing import BatchDataProcessing

from src.train_pipeline import mdpmodel_train, mdpmodel_test
import os
from time import time
import pandas as pd
from sklearn.metrics import mean_squared_error

# avoid bug, ref:https://github.com/dmlc/xgboost/issues/1715
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

'''change in 01/06/2020, now input data is log(1+r), in the prev version, all rewards are transformed back to real scale for evaluation,
now use log(1+r) for evaluation, delete the expm1 part'''

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--file_num", type=int, default=1)
parser.add_argument("--policy", type=str)
parser.add_argument("--N", type=int)
parser.add_argument("--train_num_traj", type=int)
parser.add_argument("--dev_num_traj", type=int, default=50)
parser.add_argument("--scenario", type=str, default='USEBATCH')  # SIMULATOR USEBATCH
parser.add_argument("--train_batch_size", type=int)
parser.add_argument("--train_num_batches", type=int)

args = parser.parse_args()

if args.train_batch_size:
    live_config.train_batch_size = args.train_batch_size

if args.train_num_batches:
    live_config.train_num_batches = args.train_num_batches

print(args)


def generate_data(bdp_original, config, bpl_eval=None):
    t0 = time()
    memory = SampleSet(config)
    dev_memory = SampleSet(config)
    traj_set = TrajectorySet(config)
    scores = deque()
    # Gaomin: change_padding from 0 to 50 12/30/19
    actions_array = np.zeros((config.sample_num_traj, config.max_length)) + 50
    # actions_array = np.zeros((config.sample_num_traj, config.max_length))
    p_ie_array = np.zeros((config.sample_num_traj, config.max_length))
    p_ib_array = np.zeros((config.sample_num_traj, config.max_length))
    isweight_array = np.zeros((config.sample_num_traj, config.max_length))

    # Check sample_num_traj, should be args.train_num_traj + args.dev_num_traj
    print("sample_num_traj", config.sample_num_traj)

    for i_episode in range(config.sample_num_traj):

        if i_episode % 10 == 0:
            print("{} trajectories generated".format(i_episode))
        episode = bdp_original.run_episode(i_episode, eps=config.behavior_eps, track=True)
        done = False
        n_steps = 0  # n_step: state-action-reward pair in trajectory
        factual = 1
        traj_set.new_traj()

        while not done:
            action = episode[n_steps][1]  # changed on Oct 1, 2019
            if bpl_eval:
                # action is from real data
                # eval_policy output probability of all actions given states(from real data)
                # p_pie: probability from eval_policy of (s_t, a_t)
                p_pie = bpl_eval.policy(episode[n_steps][0], eps=0)[action]
            else:
                # use behavior policy as eval_policy
                p_pie = bdp_original.bpolicy(episode[n_steps][0])[action]

            # As real data is random-select-action policy, behavior policy is frequency of action from real data
            p_pib = bdp_original.bpolicy(episode[n_steps][0])[action]

            p_pie = FloatTensor([p_pie])
            p_pib = FloatTensor([p_pib])
            isweight = p_pie / p_pib

            actions_array[i_episode, n_steps] = action
            p_ib_array[i_episode, n_steps] = p_pib
            p_ie_array[i_episode, n_steps] = p_pie
            isweight_array[i_episode, n_steps] = isweight

            last_factual = factual * (1 - p_pie)
            factual = factual * p_pie
            # changed on 12/03/2019
            episode[n_steps][0] = episode[n_steps][0].astype(float)
            episode[n_steps][3] = episode[n_steps][3].astype(float)
            state = preprocess_state(episode[n_steps][0], config.state_dim)
            next_state = preprocess_state(episode[n_steps][3], config.state_dim)
            reward = episode[n_steps][2]
            reward = preprocess_reward(reward)
            action = preprocess_action(action)
            done = (n_steps == len(episode) - 1) or (n_steps == config.max_length - 1)

            if i_episode < config.train_num_traj:
                memory.push(state, action, next_state, reward, done,
                            isweight, None, n_steps, factual, last_factual, None, None, None, None, None)
            else:
                dev_memory.push(state, action, next_state, reward, done,
                                isweight, None, n_steps, factual, last_factual, None, None, None, None, None)

            traj_set.push(state, action, next_state, reward, done, isweight, None,
                          n_steps, factual, last_factual, None, None, None, None, None)
            n_steps += 1
    memory.flatten()  # prepare flatten data
    dev_memory.flatten()
    memory.update_u()  # prepare u_{0:t}
    dev_memory.update_u()
    t1 = time()
    print("Generating {} trajectories took {} minutes".format(
        config.sample_num_traj, (t1 - t0) / 60))

    return memory, dev_memory, traj_set, scores


def train_model(memory, dev_memory, config, loss_mode):
    mdpnet = MDPnet(config)

    best_train_loss = 100 # An extreme large number for loss for lr decay: if train loss is not decreasing decay.
    lr = config.lr
    train_loss_list = []
    dev_loss_list = []
    # changed 12/04/2019
    print('-------mdpnet training size=', config.train_num_traj)
    for i_episode in range(config.train_num_traj):
        train_loss = 0
        dev_loss = 0
        optimizer = optim.Adam(mdpnet.parameters(), lr=lr)
        for i_batch in range(config.train_num_batches):
            train_loss_batch = mdpmodel_train(memory, mdpnet, optimizer,
                                              loss_mode, config)
            train_loss = ((train_loss * i_batch + train_loss_batch)
                          / (i_batch + 1))
            if config.dev_num_traj > 0:
                dev_loss_batch = mdpmodel_test(dev_memory, mdpnet, 0,
                                               config)
                dev_loss = (dev_loss * i_batch + dev_loss_batch) / (i_batch + 1)
        if (i_episode + 1) % config.print_per_epi == 0:
            print('Episode {}: train loss {:.3e}, dev loss {:.3e}'.format(
                i_episode + 1, train_loss, dev_loss))
        # learning rate decay condition
        if train_loss < best_train_loss:
            best_train_loss = train_loss
        else:
            lr *= config.lr_decay
        # Records train_loss and dev_loss for each episode.
        train_loss_list += [train_loss]
        dev_loss_list += [dev_loss]

    return train_loss_list, dev_loss_list, mdpnet


def rollout_batch(bpl_eval, config, init_states, mdpnet, done_model, num_rollout,
                  init_done=None, init_actions=None):
    ori_batch_size = init_states.size()[0]  # = eval_pib_num_rollout
    batch_size = init_states.size()[0] * num_rollout  # = eval_pib_num_rollout*eval_num_rollout
    init_states = init_states.repeat(num_rollout, 1)

    if init_actions is not None:
        init_actions = init_actions.repeat(num_rollout, 1)
    if init_done is not None:
        init_done = init_done.repeat(num_rollout)
    states = init_states
    if init_done is None:
        done = ByteTensor(batch_size)
        done.fill_(0)
    else:
        done = init_done.byte()

    if init_actions is None:
        # choose action seleced by eval_policy as init_actions
        actions = []
        for i_actions in range(batch_size):
            action = bpl_eval.policy(states[i_actions], eps=0)
            action = action / np.sum(action)  # dirty change
            actions.append(np.random.multinomial(1, action).argmax())  # Oct 1,2019
    else:
        actions = init_actions

    n_steps = 0
    t_reward = torch.zeros(batch_size)
    # This finish flag would track the process to run till n_steps reach max_length
    finish = False
    while not finish:
        if n_steps > 0:
            actions = []
            for i_actions in range(batch_size):
                # get reward from eval_policy
                action = bpl_eval.policy(states[i_actions], eps=0) # prob of each action given state
                action = action / np.sum(action)  # normalize action to prob of each action
                actions.append(np.random.multinomial(1, action).argmax())  # Oct 1, 2019

        states = Variable(Tensor(states))
        states_diff, reward, _ = mdpnet.forward(states.cpu())
        # Debug start
        # check all reward from mdpnet
        # reward_mdp = torch.expm1(reward).detach().numpy()
        # pd.DataFrame(reward_mdp).to_csv('reward_mdp_{}.csv'.format(n_steps))
        # Debug end

        states_diff = states_diff.data
        actions = LongTensor(actions)

        # get the reward of action suggested by eval policy
        reward = reward.data.gather(1, actions.cpu().view(-1, 1)).squeeze()
        expanded_actions = actions.view(-1, 1).unsqueeze(2)
        expanded_actions = expanded_actions.expand(-1, -1, config.state_dim)
        states_diff = states_diff.gather(1, expanded_actions.cpu()).squeeze()
        # state_diff = state_diff.view(-1, config.state_dim)

        next_states = states_diff + states.data
        states = next_states    # states transition given by MDPnet
        # reward = torch.expm1(reward)    # transform log reward back to real data scale
        t_reward = t_reward + config.gamma ** n_steps * torch.mul(reward,
                                                                  (1 - done))  # choose reward that only done = not done

        done_next = torch.tensor(done_model.forward(states.cpu(), actions.cpu()))
        done_next[done.long().nonzero()] = 1  # use done as mask # check if done_t is true or not
        done = done_next.byte()  # update done

        finish = n_steps == config.max_length - 1
        n_steps += 1

    value = t_reward.numpy()
    value = np.reshape(value, [num_rollout, ori_batch_size])

    
    return np.mean(value, 0)


def compute_values(bpl_eval, traj_set, model, config, done_model, model_type='MDP'):
    # use the whole traj set to compute V, Q
    num_samples = len(traj_set)
    # num_samples = config.num_samples
    traj_len = np.zeros(num_samples, 'int')
    state_tensor = FloatTensor(num_samples, config.max_length, config.state_dim).zero_()
    action_tensor = LongTensor(num_samples, config.max_length, 1).zero_()
    done_tensor = ByteTensor(num_samples, config.max_length).fill_(1)
    V_value = np.zeros((num_samples, config.max_length))
    Q_value = np.zeros((num_samples, config.max_length))

    for i_traj in range(num_samples):
        traj_len[i_traj] = len(traj_set.trajectories[i_traj])
        state_tensor[i_traj, 0:traj_len[i_traj], :] = torch.cat([t.state for t in traj_set.trajectories[i_traj]])
        action_tensor[i_traj, 0:traj_len[i_traj], :] = torch.cat([t.action for t in traj_set.trajectories[i_traj]])
        done_tensor[i_traj, 0:traj_len[i_traj]].fill_(0)
        if traj_len[i_traj] < config.max_length:
            done_tensor[i_traj, traj_len[i_traj]:].fill_(1)

    for i_step in range(config.max_length):
        t0 = time()
        if model_type == 'MDP':
            # V_value of using s_i as initial state [not consider prev steps/length of traj] and use eval policy to select all actions
            V_value[:, i_step] = rollout_batch(bpl_eval=bpl_eval, init_states=state_tensor[:, i_step, :], mdpnet=model,
                                               done_model=done_model,
                                               num_rollout=config.eval_num_rollout, config=config,
                                               init_done=done_tensor[:, i_step])
            print('process:' + str(i_step) + '/' + str(config.max_length))
            # Q value would be given initialze actions but the afterwards actions are still selected from evaluation policy
            Q_value[:, i_step] = rollout_batch(bpl_eval=bpl_eval, init_states=state_tensor[:, i_step, :], mdpnet=model,
                                               done_model=done_model,
                                               num_rollout=config.eval_num_rollout, config=config,
                                               init_done=done_tensor[:, i_step],
                                               init_actions=action_tensor[:, i_step, :])
        elif model_type == 'IS':
            pass
        t1 = time()
        print("Compute values for {} steps took {} minutes".format(i_step, (t1 - t0) / 60))
    return V_value, Q_value


def single_step_mdp(config, states, actions, mdpnet, i_step, init_done=None):
    batch_size = states.size()[0]

    if init_done is None:
        done = ByteTensor(batch_size)
        done.fill_(0)
    else:
        done = init_done.byte()

    states = Variable(Tensor(states))
    states_diff, reward, _ = mdpnet.forward(states.cpu())
    # add done flag to mdp reward - MDP reward is used to eval MDP prediction

    # reward = torch.expm1(reward) # transform log reward back to real data scale
    reward_mdp = torch.mul(torch.transpose(reward, 0, 1), (1 - done)).detach().numpy()
    reward_mdp = reward_mdp.T
    reward_selected = reward.gather(1, actions.cpu()).squeeze()
    reward_selected_done = torch.mul(reward_selected, (1 - done))

    return np.max(reward_mdp, axis=1), np.mean(reward_mdp, axis=1), np.median(reward_mdp, axis=1), \
           reward_selected.detach().numpy(), reward_selected_done.detach().numpy()


def evaluate_mdpmodel(traj_set, model, config):
    num_samples = len(traj_set)
    traj_len = np.zeros(num_samples, 'int')
    state_tensor = FloatTensor(num_samples, config.max_length, config.state_dim).zero_()
    action_tensor = LongTensor(num_samples, config.max_length, 1).zero_()
    done_tensor = ByteTensor(num_samples, config.max_length).fill_(1)

    # Debug start
    # intermediate result of MDP net reward
    pred_reward_max = np.zeros((num_samples, config.max_length))
    pred_reward_mean = np.zeros((num_samples, config.max_length))
    pred_reward_median = np.zeros((num_samples, config.max_length))
    pred_reward_selected = np.zeros((num_samples, config.max_length))
    pred_reward_selected_done = np.zeros((num_samples, config.max_length))
    # Debug end

    for i_traj in range(num_samples):
        traj_len[i_traj] = len(traj_set.trajectories[i_traj])
        state_tensor[i_traj, 0:traj_len[i_traj], :] = torch.cat([t.state for t in traj_set.trajectories[i_traj]])
        action_tensor[i_traj, 0:traj_len[i_traj], :] = torch.cat([t.action for t in traj_set.trajectories[i_traj]])
        done_tensor[i_traj, 0:traj_len[i_traj]].fill_(0)
        if traj_len[i_traj] < config.max_length:
            done_tensor[i_traj, traj_len[i_traj]:].fill_(1)

    for i_step in range(config.max_length):
        max_reward, mean_reward, median_reward, \
        reward_selected, reward_selected_done = single_step_mdp(config=config,
                                                                states=state_tensor[:, i_step, :],
                                                                actions=action_tensor[:, i_step,:],
                                                                mdpnet=model, init_done=done_tensor[:,i_step],
                                                                i_step=i_step)
        pred_reward_max[:, i_step] = max_reward
        pred_reward_mean[:, i_step] = mean_reward
        pred_reward_median[:, i_step] = median_reward
        pred_reward_selected[:, i_step] = reward_selected
        pred_reward_selected_done[:, i_step] = reward_selected_done

        print('Eval MDP process process:' + str(i_step) + '/' + str(config.max_length))

    # Debug start
#     pd.DataFrame(pred_reward_max).to_csv('pred_reward_max.csv')
#     pd.DataFrame(pred_reward_mean).to_csv('pred_reward_mean.csv')
#     pd.DataFrame(pred_reward_median).to_csv('pred_reward_median.csv')
#     pd.DataFrame(pred_reward_selected).to_csv('pred_reward_selected.csv')
#     pd.DataFrame(pred_reward_selected_done).to_csv('pred_reward_selected_done.csv')
    # Debug end

    return pred_reward_selected


def doubly_robust2(traj_set, V_value, Q_value, config, wis=False):  # changed on Sep 24, 2019
    num_samples = len(traj_set)
    # num_samples = config.num_samples
    weights = np.zeros((num_samples, config.max_length))
    weights_sum = np.zeros(config.max_length)
    for i_traj in range(num_samples):
        for n in range(config.max_length):
            if n >= len(traj_set.trajectories[i_traj]):
                weights[i_traj:, n] = weights[i_traj, n - 1]
                break
            # changed on 12/26/2019 Gaomin debug: PSIS & IS difference
            if n == 0:
                weights[i_traj, n] = traj_set.trajectories[i_traj][n].isweight[0].item()
            else:
                weights[i_traj, n] = weights[i_traj, n - 1] * traj_set.trajectories[i_traj][n].isweight[0].item()
    # Debug start
#     pd.DataFrame(weights).to_csv('DBR_weight_{}.csv'.format(args.policy))
    # Debug end

    # Normalized weight
    if wis:
        for n in range(config.max_length):
            # accumulate sum of weights
            weights_sum[n] = np.sum(weights[:, n])
            if weights_sum[n] != 0:
                weights[:, n] = (weights[:, n] * num_samples) / weights_sum[n]
    # Debug start
#     pd.DataFrame(weights).to_csv('DBR_weight_w_{}.csv'.format(args.policy))
    # Debug end

    value = np.zeros(num_samples)
    for i_traj in range(num_samples):
        w = 1
        for t in traj_set.trajectories[i_traj]:
            # reward = (np.exp(t.reward[0].item()) - 1) # transform log reward back to real data scale
            reward = t.reward[0].item()
            value[i_traj] += (config.gamma ** t.time) *\
                             (weights[i_traj, t.time] * (reward - Q_value[i_traj, t.time])
                                + w *V_value[i_traj, t.time])
            # changed on 12/25/19 Revise w in compute_value into weight_t-1.
            w = weights[i_traj, t.time]
            if w == 0:
                break
    return value

def importance_sampling2(traj_set, wis=False):
    num_samples = len(traj_set)
    # num_samples = config.num_samples
    value = np.zeros(num_samples)
    weights = np.zeros(num_samples)

    for i_traj in range(num_samples):
        l = len(traj_set.trajectories[i_traj])
        tmp = 1
        for n in range(l):
            # w is to break the loop
            w = traj_set.trajectories[i_traj][n].isweight[0].item()
            if w == 0:  # changed on Sep 24, 2019
                break
            tmp *= traj_set.trajectories[i_traj][n].isweight[0].item()
        weights[i_traj] = tmp
    # Debug start
#     pd.DataFrame(weights).to_csv('importance_sampling2_{}.csv'.format(args.policy))
    # Debug end

    # Normalize weights
    if wis:
        weights = (weights * num_samples) / np.sum(weights)
    # weights are calculated on the whole trajectory

    # Debug start
#     pd.DataFrame(weights).to_csv('importance_sampling2_w_{}.csv'.format(args.policy))
    # Debug end

    for i_traj in range(num_samples):
        for t in traj_set.trajectories[i_traj]:
            # reward = (np.exp(t.reward[0].item()) - 1) # transform log reward back to real data scale
            reward = t.reward[0].item()
            value[i_traj] += config.gamma ** t.time * weights[i_traj] * reward

    return value


if __name__ == "__main__":
    t0 = time()

    config = live_config
    if args.train_num_traj:
        config.train_num_traj = args.train_num_traj
        config.dev_num_traj = args.dev_num_traj
        config.sample_num_traj = args.train_num_traj + args.dev_num_traj
    else:
        config.sample_num_traj = config.train_num_traj + config.dev_num_traj
    if config.train_batch_size > config.train_num_traj:
        config.train_batch_size = config.train_num_traj

    """ Load live environment - the environment comes with a policy which can be
    made eps greedy. """
    with open('./live_domain/live_simulator/live_preset_hidden_params', 'rb') as f:
        preset_hidden_params = pickle.load(f, encoding='latin1')

    print('---running bdp_original')
    bdp_original = BatchDataProcessing(scenario=args.scenario, eval=False, upsample=False)  # create original batch data
    print('---running bpl_eval')
    bpl_eval = BatchPolicyLearning(scenario=args.scenario, ModelPolicy=args.policy)  # eval_policy
    print('---running bdp_upsample')
    bdp_upsample = BatchDataProcessing(scenario=args.scenario, eval=False, upsample=True) # upsampled batch data to train MDPnet

    if args.N:
        config.N = args.N

    # Add dict to save results 2020/01/04
    result_dict = {}
    for i in range(config.N):
        # upsample data for training MDP
        print('Generate upsample data for MDPnet training')
        memory, dev_memory, traj_set, scores = generate_data(bpl_eval=bpl_eval, bdp_original=bdp_upsample, config=config)

        # Debug start
        # Gaomin: save true rewards of upsample data
        reward_array = np.zeros((config.sample_num_traj, config.max_length))
        # calculate traj_set average
        for i_traj in range(config.sample_num_traj):
            for t in traj_set.trajectories[i_traj]:
                # reward = (np.exp(t.reward[0].item()) - 1) # transform log reward back to real data scale
                reward = t.reward[0].item()
                reward_array[i_traj, t.time] = reward
#         pd.DataFrame(reward_array).to_csv('trajectories_true_rewards_upsample.csv')
        # Debug end

        print('Learn our mdp model')
        _, _, mdpnet = (
            train_model(memory, dev_memory, config, 1))
        print('Learn the baseline mdp model')
        _, _, mdpnet_unweight = (
            train_model(memory, dev_memory, config, 0))

        print('Learn the done prediction model')
        done_model = DonePredict(n_estimators=100, max_depth=3)
        done_model.train(traj_set)

        print('Evaluate models using evaluation policy on the same initial states')
        # Evaluation mdpnet
        mdpnet.eval()
        mdpnet_unweight.eval()
        # use all states in traj_set to eval mdpnet
        pred_reward_selected = evaluate_mdpmodel(traj_set, mdpnet, config)
        print('=' * 50)
        rmse_model = np.sqrt(mean_squared_error(pred_reward_selected, reward_array))
        print('=' * 20, "root_mean_squared_error:", rmse_model, '=' * 20)
        print('=' * 50)
        mean_arr = np.zeros((config.sample_num_traj, config.max_length)) + np.mean(reward_array)
        rmse_mean = np.sqrt(mean_squared_error(mean_arr, reward_array))
        print('=' * 20, "mean_reward:", mean_arr[0][0], '=' * 20)
        print('=' * 20, "rmse_mean:", rmse_mean, '=' * 20)
        r_square = 1 - rmse_model / rmse_mean
        print('=' * 20, "r_square:", r_square, '=' * 20)
        print('=' * 50)
        result_dict["MDP_rsquared"] = r_square

        ########################################################
        # for estimator use data without upsample
        print('Generate original data for estimator')
        memory, dev_memory, traj_set, scores = generate_data(bpl_eval=bpl_eval,
                                                             bdp_original=bdp_original, config=config)


        # Debug start
        # Gaomin: save true rewards of original data
        reward_array = np.zeros((config.sample_num_traj, config.max_length))
        # calculate traj_set average
        average_traj_len =[]
        for i_traj in range(config.sample_num_traj):
            average_traj_len.append(len(traj_set.trajectories[i_traj]))

            for t in traj_set.trajectories[i_traj]:
                # reward = (np.exp(t.reward[0].item()) - 1) # transform log reward back to real data scale
                reward = t.reward[0].item()
                reward_array[i_traj, t.time] = reward
#         pd.DataFrame(reward_array).to_csv('trajectories_true_rewards_original.csv')
        print(np.mean(average_traj_len))
        # Debug end


        print('start estm')
        # use bdp original as data
        # Changed @ 01/03/2020 Gaomin
        # Generate initial state for each trajactory as init_states
        init_states = []
        num_samples = len(traj_set)
        for i_traj in range(num_samples):
            init_states.append(traj_set.trajectories[i_traj][0].state)
        init_states = torch.cat(init_states)

        estm = rollout_batch(bpl_eval, config, init_states, mdpnet, done_model,
                             config.eval_num_rollout)

        estm_bsl = rollout_batch(bpl_eval, config, init_states, mdpnet_unweight, done_model,
                                 config.eval_num_rollout)

        # use traj_set as data
        print('start wdr')
        V, Q = compute_values(bpl_eval, traj_set, mdpnet, config, done_model=done_model, model_type='MDP')
#         pd.DataFrame(V).to_csv("V_value_wdr_{}.csv".format(args.policy))
#         pd.DataFrame(Q).to_csv("Q_value_wdr_{}.csv".format(args.policy))
        
        wdr2 = doubly_robust2(traj_set, V, Q, config, wis=True)  # changed on Nov 22, 2019
        print('---wdr2.mean', np.mean(np.array(wdr2)))

        print('start wdr baseline')
        V, Q = compute_values(bpl_eval, traj_set, mdpnet_unweight, config, done_model=done_model, model_type='MDP')
        wdr_bsl2 = doubly_robust2(traj_set, V, Q, config, wis=True)  # changed on Nov 22, 2019

        print('start ips')
        V, Q = compute_values(bpl_eval, traj_set, None, config, done_model=done_model, model_type='IS')
        # fro importance sampling V and Q are 0.
        ips2 = importance_sampling2(traj_set)  # changed on Sep 24, 2019
        pdis2 = doubly_robust2(traj_set, V, Q, config, wis=False)  # changed on Sep 24, 2019
        wpdis2 = doubly_robust2(traj_set, V, Q, config, wis=True)  # changed on Sep 24, 2019

        print("RepBM MDP model mean:", np.mean(estm))
        print("MDP model mean:", np.mean(estm_bsl))
        print("WDR model2 mean:", np.mean(wdr2))
        print("WDR MDP model2 mean:", np.mean(wdr_bsl2))
        print("IS2 mean:", np.mean(ips2))
        print("PSIS2 mean:", np.mean(pdis2))
        print("WPSIS2 mean:", np.mean(wpdis2))

        print("RepBM MDP median:", np.median(estm))
        print("MDP median:", np.median(estm_bsl))
        print("WDR model2 median:", np.median(wdr2))
        print("WDR MDP model2 median:", np.median(wdr_bsl2))
        print("IS2 median:", np.median(ips2))
        print("PSIS2 median:", np.median(pdis2))
        print("WPSIS2 median:", np.median(wpdis2))

        result_dict["RepBM MDP model mean"] = np.mean(estm)
        result_dict["MDP model mean"] = np.mean(estm_bsl)
        result_dict["WDR model2 mean"] = np.mean(wdr2)
        result_dict["WDR MDP model2 mean"] = np.mean(wdr_bsl2)
        result_dict["IS2 mean"] = np.mean(ips2)
        result_dict["PSIS2 mean"] = np.mean(pdis2)
        result_dict["WPSIS2 mean"] = np.mean(wpdis2)
        result_dict["RepBM MDP model median"] = np.median(estm)
        result_dict["MDP model median"] = np.median(estm_bsl)
        result_dict["WDR model2 median"] = np.median(wdr2)
        result_dict["*WDR MDP model2 median"] = np.median(wdr_bsl2)
        result_dict["IS2 median"] = np.median(ips2)
        result_dict["PSIS2 median"] = np.median(pdis2)
        result_dict["WPSIS2 median"] = np.median(wpdis2)

    pd.Series(result_dict).to_frame().to_csv("result_evaluator_model_{}.csv".format(args.policy))
    t1 = time()
    print("\nEvaluator \n took {} minutes".format(
        (t1 - t0) / 60))

