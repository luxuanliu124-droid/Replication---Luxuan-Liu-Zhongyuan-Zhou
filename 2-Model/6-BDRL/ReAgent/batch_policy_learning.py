import pickle
import numpy as np
from src.config import live_config
import sys 
import gc 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import re
import os


import sys
from ml.rl.json_serialize import from_json
from ml.rl.workflow.base_workflow import BaseWorkflow
from ml.rl.evaluation.evaluator import Evaluator
import torch
import os 
from ml.rl.workflow.transitional import create_dqn_trainer_from_params
from ml.rl.parameters import (
    DiscreteActionModelParameters,
    EvaluationParameters,
    NormalizationParameters,
    RainbowDQNParameters,
    RLParameters,
    TrainingParameters,
)
from ml.rl.workflow.helpers import (
    minibatch_size_multiplier,
    parse_args,
    update_model_for_warm_start
)
from ml.rl.models.dqn import FullyConnectedDQN
from ml.rl.training.dqn_trainer import DQNTrainer
from ml.rl.training.parametric_dqn_trainer import ParametricDQNTrainer

import numpy as np

import sys
sys.path.append("/home/xiao/rl_lib/ReAgent/")

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class BCQ_model():
    def __init__(self):
        self.load_model()
    
    def load_model(self):
        params = parse_args(["","-p","/home/xiao/rl_lib/ReAgent/ml/rl/workflow/sample_configs/discrete_action/dqn_example_123119.json"])
        params["training"]["minibatch_size"] *= minibatch_size_multiplier(
            params["use_gpu"], params["use_all_avail_gpus"]
        )
        action_names = params["actions"]
        rl_parameters = from_json(params["rl"], RLParameters)
        training_parameters = from_json(params["training"], TrainingParameters)
        rainbow_parameters = from_json(params["rainbow"], RainbowDQNParameters)
        if "evaluation" in params:
            evaluation_parameters = from_json(params["evaluation"], EvaluationParameters)
        else:
            evaluation_parameters = EvaluationParameters()
        model_params = DiscreteActionModelParameters(
            actions=action_names,
            rl=rl_parameters,
            training=training_parameters,
            rainbow=rainbow_parameters,
            evaluation=evaluation_parameters,
        )
        state_normalization = BaseWorkflow.read_norm_file("/home/xiao/rl_lib/ReAgent/"+params["state_norm_data_path"])
        model = create_dqn_trainer_from_params(
            model_params,
            state_normalization,
        )
        state = torch.load("/home/xiao/rl_lib/ReAgent/outputs17/trainer_1578219871.pt")
        if isinstance(model, DQNTrainer):
            state["q_network"] = type(state["q_network"])(
                ("{}".format(k).replace("fc_dqn.",""), v) for k, v in state["q_network"].items() if "data_parallel.module." not in k 
            )
            model.q_network.load_state_dict(state["q_network"])
            model.q_network_target.load_state_dict(state["q_network_target"])
            model.q_network_optimizer.load_state_dict(state["q_network_optimizer"])
        self.model = model
   
    def select_action(self, state):
        action = self.model.internal_prediction(torch.Tensor(state))
        print("Actions:", action)
        return action 



class BatchPolicyLearning(object):
    '''
    in this class, we load trained policy function
    '''
    def __init__(self, scenario='USEBATCH', ModelPolicy ='BCQ'):
        gc.disable()
        self.state = None
        self.scenario = scenario
        self.ModelPolicy = ModelPolicy

        if self.scenario == "SIMULATOR" or self.scenario == "USEBATCH":# "SIMULATOR",'USEBATCH'
            if self.ModelPolicy == 'BCQ':
                self.model = BCQ_model()
          
            
    def policy(self, state, eps=0.0):
        """Get the action under the current plan policy for the given state.

        Args:
        state: The array of state features

        Returns:
        The current greedy action under the planned policy for the given state. If no plan has been formed,
        return a random action.
        """
        if np.random.rand(1) < eps:
            return np.random.dirichlet(np.ones(self.action_dim))  # changed on Oct 1, 2019 #? 
        else:
            return self.batch_policy_learning(state)

    def batch_policy_learning(self, state):
        if self.scenario == 'SIMULATE':
            action = np.random.dirichlet(np.ones(live_config.action_size))
        else:
            try: 
                state = state.cpu().data.numpy().flatten()
            except: 
                pass
            #changed on 12/02/19 
            #state[state<0]=0   #add #
            #if np.sum(state)>1:
            #    state = state/np.sum(state) 
            action = self.model.select_action(state)
        return action

    # os.chdir('..')
    # test = BatchPolicyLearning(scenario='USEBATCH', ModelPolicy='NN')
    # sim_state = np.random.dirichlet(np.ones(live_config.state_dim))
    # test.policy(sim_state.reshape(1, -1))

