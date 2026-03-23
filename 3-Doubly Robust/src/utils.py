import os
import numpy as np
import random
import torch
from torch.autograd import Variable

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


def preprocess_state(state, input_dim):
    return Tensor(np.reshape(state, [1, input_dim]))


def preprocess_action(action):
    return LongTensor([[action]])


def preprocess_reward(reward):
    return FloatTensor([reward])


def rescale_state(state, rescale=1):
    return state*rescale


def select_action_random(action_size):
    return LongTensor([[random.randrange(action_size)]])

def epsilon_greedy_action_batch(state_tensor, qnet, epsilon, action_size):
    batch_size = state_tensor.size()[0]
    sample = np.random.random([batch_size,1])
    greedy_a = qnet.forward(
            state_tensor.type(FloatTensor)).detach().max(1)[1].view(-1, 1)
    random_a = LongTensor(np.random.random_integers(0,action_size-1,(batch_size,1)))
    return (sample < epsilon)*random_a + (sample >= epsilon)*greedy_a


def epsilon_greedy_action(state_tensor, qnet, epsilon, action_size, q_values=None):
    if q_values is None:
        q_values = qnet.forward(Variable(state_tensor, volatile=True).type(FloatTensor)).detach()
    sample = random.random()
    if sample > epsilon:
        return q_values.max(1)[1].view(-1, 1)
    else:
        return LongTensor([[random.randrange(action_size)]])


def epsilon_greedy_action_prob(state_tensor, qnet, epsilon, action_size, q_values = None):
    if q_values is None:
        q_values = qnet.forward(Variable(state_tensor, volatile=True).type(FloatTensor)).detach()
    max_action = q_values.max(1)[1].view(1, 1).item()
    prob = FloatTensor(1,action_size)
    prob.fill_(epsilon/action_size)
    prob[0,max_action] = 1-epsilon+epsilon/action_size
    return prob


def restrict_state_region(state_tensor):
    flag = abs(state_tensor[0,0].item()) < 2.2 and abs(state_tensor[0,2].item()) < 0.2
    return flag


def learning_rate_update(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def weighted_mse_loss(input, target, weights):
    out = ((input-target)**2)
    if len(list(out.size())) > 1:
        out = out.mean(1)
    out = out * weights #.expand_as(out)
    loss = out.mean(0)
    return loss


def mmd_lin(rep1, rep2, p=0.5):
    mean1 = rep1.mean(0)
    mean2 = rep2.mean(0)
    mmd = ((2.0*p*mean1 - 2.0*(1.0-p)*mean2)**2).sum(0)
    return mmd

