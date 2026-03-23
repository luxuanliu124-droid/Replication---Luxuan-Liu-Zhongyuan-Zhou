import sys
import pandas as pd
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

print(os.listdir("outputs031620/"))


##############################################
#### CHANGE THE CONFIG PATH IN SYS.CONFIG #### 
############################################## 
# this config file and model path must match! 
# Ex: 
#params = parse_args(["ml/rl/workflow/sample_configs/discrete_action/dqn_example_123119.json"])

print(sys.argv)
params = parse_args(sys.argv)
print(params)

# Running Command: python load_trained_model.py  -p ml/rl/workflow/sample_configs/discrete_action/dqn_example_123119.json

######################################
#### LOADING THE CONFIG FILES ... #### 
######################################
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

######################################
#### CREATE MODEL PARAMETERS  ... #### 
######################################
model_params = DiscreteActionModelParameters(
    actions=action_names,
    rl=rl_parameters,
    training=training_parameters,
    rainbow=rainbow_parameters,
    evaluation=evaluation_parameters,
)
state_normalization = BaseWorkflow.read_norm_file(params["state_norm_data_path"])

######################################
#### CREATE A MODEL  ...          #### 
######################################
model = create_dqn_trainer_from_params(
            model_params,
            state_normalization,
        )

######################################
#### LOAD TRAINED STATE AND LOAD IT TO MODEL ... #### 
######################################
### Change the model  path here 
state = torch.load("outputs031620/trainer_1584393818.pt")
# model = update_model_for_warm_start(model, "outputs/trainer_1575344567.pt")
if isinstance(model, DQNTrainer):
    state["q_network"] = type(state["q_network"])(
        ("{}".format(k).replace("fc_dqn.",""), v) for k, v in state["q_network"].items() if "data_parallel.module." not in k 
    )
    model.q_network.load_state_dict(state["q_network"])
    model.q_network_target.load_state_dict(state["q_network_target"])
    model.q_network_optimizer.load_state_dict(state["q_network_optimizer"])
    
    
print(dir(model))  
def get_action(model, state):
    ### Customize this funciton 
    action = model.internal_prediction(state)
    print("Actions:", action)
    return action 

def get_reward(model, state, action):
    reward = model.internal_reward_estimation(state)
    print("Reward:", reward)
    return reward

def select_action(model, state):
    state = torch.Tensor(state.reshape(-1))
    action = get_action(model, state )
    return np.argmax(action.numpy())
    

# for _ in range(10):
#     state = torch.Tensor( np.random.random(375).reshape(-1))
#     print(select_action(model, state))
#     action = get_action(model, state)
#     reward = get_reward(model, state, action)

####### read real data
file = "~/rl_lib/ReAgent/data031620/cartpole_discrete_timeline.json"
data = pd.read_json(file,lines=True)

data["state"] = data.state_features.apply(lambda x: np.array(list(x.values())))
data_need = data[["mdp_id","sequence_number","state"]]
data_need["BCQ_Action"] = data_need["state"].apply(lambda x: select_action(model=model, state=x))
data_need.to_csv("BCQ_distribution031620.csv")
        
        
        
        
