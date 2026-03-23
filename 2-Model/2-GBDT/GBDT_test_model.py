import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor,GradientBoostingClassifier 
import pickle
import pandas as pd
from hyperparameter import live_config
import os
from time import time


if __name__ == "__main__":
    
    num_state_var = 375 
    num_action = 25     

    pretrained_model_path = os.path.join('.',live_config.filename_gbdt+'_regression.sav')

    print(pretrained_model_path)
    model_dict = pickle.load(open(pretrained_model_path, 'rb'))
    model = model_dict['model']
    num_state_var = model_dict['num_state_var']
    num_action = model_dict['num_action']
    print('GBDT Model successfully loaded from {}'.format(pretrained_model_path))
    print('\n Hyperparameter:{}\n num_state_var:{}\n num_action:{}\n'.format(model_dict['hyperparameter'],
                                                                               num_state_var,
                                                                                   num_action))

   
    # plug in data
    data_modified = pd.read_csv('../../model_input_sample_small_test.csv')
    print('Input test data shape:',data_modified.shape)
    
    states = data_modified.iloc[:, 2:num_state_var+2]
    print('# of States:{}, States name:{}'.format(num_state_var,states.columns))
    states = states.values
    action = data_modified.iloc[:, num_state_var+2:num_state_var+2+num_action]
    print('# of Actions:{}, Actions name:{}'.format(num_action,action.columns))
    action =action.values
    rewards = data_modified.iloc[:, -1:]
    print('Rewards name:{}'.format(rewards.columns))
    rewards = rewards.values.ravel()

    X = np.concatenate([states, action], axis=1)
    y = rewards

    print("GBDT test performance R2:",model.score(x,y))

    
    print('GBDT on performance on test model: \nRsquared:',model.score(X,y))
