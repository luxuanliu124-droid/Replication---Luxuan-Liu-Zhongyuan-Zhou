# Model import（econml>=0.16 使用 orf.DMLOrthoForest + discrete_treatment=True）
from econml.orf import DMLOrthoForest
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression, LogisticRegressionCV
from econml.utilities import WeightedModelWrapper

# Helper import
import numpy as np
import pickle as pkl
import dill
import pandas as pd
from time import time
import datetime
import sys
import os

# Parameters import
from orf_hyperparameter import live_config

import sys
sys.path.append('..')
from input_file import input_file

class ORF_model(object):
    def __init__(self, input_data):
        '''
        :param input_data:
            Data of the format in each row < USERID, TIMEID, State, Action, NextState, Reward >
        '''
        self.input_data = input_data


    def train(self):
        '''
        Return: 
        trained ORF model
        '''

        # Make model inputs
        X = self.input_data[:, 2:live_config.num_state_var+2]
        #X = self.input_data[:, 2].reshape(-1, 1)
        Y = np.ravel(self.input_data[:, -1])
        T = self.input_data[:, live_config.num_state_var+2:live_config.num_state_var+live_config.num_action+2]
        T = np.where(T==1)[1].reshape(-1, 1)

        for a in [X,Y,T]:
            print(a.shape)
            print(type(a))


        # Train model（DMLOrthoForest 替代已废弃的 DiscreteTreatmentOrthoForest）
        est = DMLOrthoForest(
            n_trees=live_config.num_trees,
            max_depth=live_config.max_depth,
            min_leaf_size=live_config.min_leaf_size,
            model_Y=WeightedModelWrapper(Lasso(alpha=live_config.lambda_reg)),
            discrete_treatment=True,
        )
        est.fit(Y, T, X=X)

        # Save the model
        dill.dump(est, open(live_config.model_save_path, 'wb'))

        return est



if __name__ == "__main__":
    print("start training ORF model")
    # input_data = np.loadtxt('../data/test_data_full_102119_modified.csv', delimiter=',', skiprows=1)
    # input_data = input_data[:, 1:]
    t0 = time()
    # input_data = pd.read_csv('../../../smalldata/model_input_sample_small_train_log.csv')
    input_data = pd.read_csv(input_file)
    print("Loaded input data")
    # input_data = input_data.drop_duplicates()
    # # balance sampling
    # input_data['div_pay_or_not'] = 0
    # input_data.loc[input_data.div_pay_amt_fillna>0]['div_pay_or_not'] = 1
    # g = input_data.groupby('div_pay_or_not')
    # balance_input = g.apply(lambda x: x.sample(g.size().min())).reset_index(drop=True).values
    # print(balance_input.shape)

    # random sample（数据行数不足时用全部数据或允许有放回抽样）
    sample_num = 20000
    n_rows = len(input_data)
    if n_rows < sample_num:
        sample_num = n_rows
        replace = False
    else:
        replace = False
    small_input = input_data.sample(n=sample_num, replace=replace).values
    print("Random sample {} from input data.".format(sample_num))

    # make folder to save saved model
    today = datetime.date.today().strftime('%y%m%d')
    folder = os.path.join('../', 'saved_model_' + str(today))
    if not os.path.exists(folder):
        os.makedirs(folder)

    os.chdir(folder)
    # train and save model
    model = ORF_model(small_input)
    model.train()
    t1 = time()
    print("Generating ORF model \n took {} minutes".format(
         (t1-t0)/60))