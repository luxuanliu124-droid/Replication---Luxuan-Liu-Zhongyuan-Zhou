from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from time import time
import os
import sys
import pickle
import datetime
import sys
sys.path.append('..')
from input_file import input_file

num_state_var = 375
num_action = 25
t0 = time()

# make folder to save saved model
today = datetime.date.today().strftime('%y%m%d')
folder = os.path.join('../', 'saved_model_'+str(today))
if not os.path.exists(folder):
    os.makedirs(folder)

# plug in data
# data_modified = pd.read_csv('../../../smalldata/model_input_sample_small_train_log.csv')
data_modified = pd.read_csv(input_file)
print('Input data shape:',data_modified.shape)
print('num_state_var:',num_state_var)
print('num_action:',num_action)

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

os.chdir(folder)
model = LinearRegression(fit_intercept=False)
# train and save model
model.fit(X, y)
print('Training R2',model.score(X, y))
t1 = time()
print("Training OLS model \n took {} minutes".format(
     (t1-t0)/60))

file_name = 'LinearRegression.sav'
pickle.dump(model, open(file_name, 'wb'))



