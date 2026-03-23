import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor,GradientBoostingClassifier 
import pickle as pkl
import pandas as pd
from hyperparameter import live_config
import argparse
import datetime
from time import time
import os

import sys
sys.path.append('..')
from input_file import input_file

# # deal with argument parser 
parser = argparse.ArgumentParser()
parser.add_argument("--prediction", type=str, default='regression')
args = parser.parse_args()
prediction = args.prediction

class gbdt_model(object):
    def __init__(self, num_state_var, num_action):
        '''

        :param num_state_var: int
            Number of state variables
        :param num_action: int
            Number of action types
        '''
        self.num_state_var = num_state_var
        self.num_action = num_action


    def train(self, states, action, rewards, param_grid, prediction,file_name):
        '''
        Data be train+validation dataset
        Use gridseachCV to search on the best hyperparameters and return model

        :param states: array-like, shape (n_samples, num_state_var)
                       states features
        :param action: array-like, shape (n_samples, num_action)
                       action features
        :param rewards: array-like, shape (n_samples)
                        rewards target
        :param param_grid: dict (from live_config)
                           hyperparameter for gridseach to search on
        :param prediction: type of prediction to make

        :return: the best GBDT model and save
        '''
        # check the feature dimension
        assert states.shape[1] == self.num_state_var
        assert action.shape[1] == self.num_action
        # concat features
        X = np.concatenate([states, action], axis=1)
        y = rewards

        if prediction == 'regression':

        	gd_sr = GridSearchCV(estimator=GradientBoostingRegressor(),
                             param_grid=param_grid,
                             cv=3,
                             refit=True,
                             return_train_score=True,
                             verbose=10,
                             n_jobs=-1)
        else:
        	gd_sr = GridSearchCV(estimator=GradientBoostingClassifier(),
                             param_grid=param_grid,
                             cv=3,
                             refit=True,
                             return_train_score=True,
                             verbose=10,
                             n_jobs=-1)


        gd_sr.fit(X, y)
        # look at the best estimator that was found by GridSearchCV
        print ("Best Estimator learned through GridSearch")
        print(gd_sr.best_estimator_)
        print("\nBest score from best Estimator")
        print(gd_sr.best_score_)
        print("\nGrid scores on development set:")
        means = gd_sr.cv_results_['mean_test_score']
        stds = gd_sr.cv_results_['std_test_score']
        means_train = gd_sr.cv_results_['mean_train_score']

        for mean, std, means_train, params in zip(means, stds, means_train,gd_sr.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r with train score%0.3f"
                  % (mean, std * 2, params, means_train))

        # save the model
        # save a dict to do sanity check
        savedict = {
            'model' : gd_sr.best_estimator_,
            'hyperparameter': gd_sr.best_params_,
            'num_state_var':self.num_state_var,
            'num_action':self.num_action
        }
        
        pkl.dump(savedict, open(file_name, 'wb'))

        return gd_sr.best_estimator_

    def train_noaction(self, states, rewards, param_grid, prediction,file_name):
    
        X = states
        y = rewards

        if prediction == 'regression':

            gd_sr = GridSearchCV(estimator=GradientBoostingRegressor(),
                             param_grid=param_grid,
                             cv=3,
                             refit=True,
                             return_train_score=True,
                             verbose=10,
                             n_jobs=-1)
        else:
            gd_sr = GridSearchCV(estimator=GradientBoostingClassifier(),
                             param_grid=param_grid,
                             cv=3,
                             refit=True,
                             return_train_score=True,
                             verbose=10,
                             n_jobs=-1)


        gd_sr.fit(X, y)
        # look at the best estimator that was found by GridSearchCV
        print ("Best Estimator learned through GridSearch")
        print(gd_sr.best_estimator_)
        print("\nBest score from best Estimator")
        print(gd_sr.best_score_)
        print("\nGrid scores on development set:")
        means = gd_sr.cv_results_['mean_test_score']
        stds = gd_sr.cv_results_['std_test_score']
        means_train = gd_sr.cv_results_['mean_train_score']

        for mean, std, means_train, params in zip(means, stds, means_train,gd_sr.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r with train score%0.3f"
                  % (mean, std * 2, params, means_train))

        # save the model
        # save a dict to do sanity check
        savedict = {
            'model' : gd_sr.best_estimator_,
            'hyperparameter': gd_sr.best_params_,
            'num_state_var':self.num_state_var,
            'num_action':self.num_action
        }
        
        pkl.dump(savedict, open(file_name, 'wb'))

        return gd_sr.best_estimator_




if __name__ == "__main__":
    
    num_state_var = 375 
    num_action = 25     
    t0 = time()

    # make folder to save saved model
    today = datetime.date.today().strftime('%y%m%d')
    folder = os.path.join('../', 'saved_model_' + str(today))
    folder = os.path.join(folder, 'gbdt_50_models')
    if not os.path.exists(folder):
        os.makedirs(folder)


    model = gbdt_model(num_state_var, num_action)
   
    # plug in data
    # read_filename = '../../../smalldata/model_input_sample_small_train_log.csv'
    read_filename = input_file
    data = pd.read_csv(read_filename)
    print('Input data shape:',data.shape)
    print('num_state_var:',model.num_state_var)
    print('num_action:',model.num_action)

    os.chdir(folder)
    action = data.iloc[:1, num_state_var+2:num_state_var+2+num_action]
    print('# of Actions:{}, Actions name:{}'.format(num_action,action.columns))
    # making subset of data
    list_data = []
    for col in list(action.columns):
        print(col)
        list_data.append(data.loc[data[col]==1])

    for i,data_modified in enumerate(list_data):
        print('------  No.{}  -------'.format(i))

        states = data_modified.iloc[:, 2:num_state_var+2]
        print('# of States:{}, States name:{}'.format(num_state_var,states.columns))
        states = states.values
        # action = data_modified.iloc[:, num_state_var+2:num_state_var+2+num_action]
        # print('# of Actions:{}, Actions name:{}'.format(num_action,action.columns))
        # action =action.values
        rewards = data_modified.iloc[:, -1:]
        print('Rewards name:{}'.format(rewards.columns))
        rewards = rewards.values.ravel()


        param_grid = live_config.param_grid
        # trained and save model in filename_gbdt
        file_name = live_config.filename_gbdt+'_model{}_.sav'.format(i)
        model.train_noaction(states, rewards, param_grid, prediction, file_name)

    t1 = time()
    print("Generating GBDT model with gridsearchCV \n took {} minutes".format(
             (t1-t0)/60))