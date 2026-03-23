#!/usr/bin/env python3

import json
import random
import sys
import time
import urllib.error
import urllib.request
from multiprocessing import Pool
from typing import Any, Dict, List, Optional, Tuple
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from scipy.stats import norm, mode
import math
import numpy as np 
import pandas as pd 
import torch
import os 
from sklearn import preprocessing
import time 
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

random.seed(0)

class Live_Env(): 


    def __init__(self):
        self.ids = 0
        self.state_dim = 375
        self.action_dim = 25
        self.plan_name = "live.json"
        self.ACTIONS = list(np.arange(25).astype("str"))


        self.data = self.read_data()
        self.state_model = self.init_state_model(self.data)
        self.reward_model, self.buy_model, self.done_model = self.init_reward_model(self.data)
        self.coupon_pair = self.read_coupon_pair()

        self.img = False
        self.t = 0
        self.done = False
        self.max_iter = 25 

        self.dict_state = None

    def reset(self):
        self.t = 0 
        self.done=False
        state, dict_state = self.get_initial_state()
        self.state = state
        self.dict_state = dict_state
        return self.state

    def step(self, action):
        
        reward = self.get_reward_from_action(self.state, action)
        updated_state = self.update_state(self.state, action)
        self.state = updated_state
        self.t+=1
        return updated_state, reward, self.t>=self.max_iter, None


    def read_coupon_pair(self):
        coupon_pair = {}
        with open("/home/xiao/rl_lib/ReAgent/coupon_pairs.csv","r") as f:
            for step, l in enumerate(f):
                l = l.rstrip("\n")
                coupon_pair[step]=float(l)
        print("coupon pairs", coupon_pair)
        return coupon_pair
    

    def update_state(self, state, action):
        old_state = state 
        '''action: one-hot list'''
        if not isinstance(action, list):
            action = action.tolist()
        
        def _get_coupon_value(self, action):
            action_idx = action.index(max(action))  # changed from 'action_idx = int(action) - 1'
            coupon_value = self.coupon_pair[action_idx]
            return coupon_value

        def _update_stationary(self, state): 
            for i in range(18):
                state[i] = norm.rvs(self.state_paras[i][0], self.state_paras[i][1])
            return state

        def _update_dynamic(self, state, action):
            coupon_value = _get_coupon_value(self, action)
            # 'buyer_dynamic_sum_man_pert_0',
            state[-5] = int( math.log( math.exp(state[-5])+coupon_value ) ) 
            # 'buyer_dynamic_avg_man_pert_0',
            state[-4] = int( math.log( ((math.exp(state[-4]))*(math.exp(state[-1]))+coupon_value ) / (math.exp(state[-1]) +1) )) 
            # 'buyer_dynamic_num_transactions',
            # 'buyer_dynamic_total_pay',
            actions = np.array(action)
            action_taken = str(action.index(max(action)) + 1)
            action = action_taken 

#             if self.buy_model.predict(np.append(state, actions).reshape(1, -1).astype(float))[0]==1:
            state[-3] = int( math.log(math.exp(state[-3])+ 1 )) 
            reward = self.get_reward_from_action(old_state, action)
            reward = 0 if reward < 0 else reward 
            state[-6] = int( math.log( math.exp(state[-3]) + reward) ) 

            # 'buyer_dynamic_num_0',
            if int(action)-1 in [5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 25, 26, 27, 28, 29, 35, 36, 37, 38, 39, 45, 46, 47, 48, 49]:
                state[-2] = int( math.log( math.exp(state[-2]) +1) ) 
            
            # 'buyer_dynamic_num_receive', 
            state[-1] = int( math.log( math.exp(state[-1])+1 ) ) 
            
            return state

        updated_state = state
        updated_state = _update_stationary(self, updated_state)
        updated_state = _update_dynamic(self, updated_state, action)

        return updated_state
  
    def read_from_csv(self, filepath):
        print(filepath)
        df = pd.read_csv(filepath)
        df["done"] = (df["user_id"]!=df["user_id"].shift(periods=-1))
        df = df[cols]
        self.mean_std = {}
        f = open("mean_std.csv","r")
        for l in f:
            l = l.rstrip("\n") 
            name, mean, std, _max = l.split(",")
            self.mean_std[name] = (float(mean), float(std), float(_max))
            self.mean_std["next_"+name] = (float(mean), float(std), float(_max))
        print(self.mean_std)
        for key in self.mean_std:
            
            df[key] = df[key]*self.mean_std[key][1] + self.mean_std[key][0] 
            df[key][df[key]>self.mean_std[key][2]] = self.mean_std[key][2]
            df[key][df[key]<0] = 0
            df[key] = np.log(df[key]+1) 
            #print(key, df[key].mean(), df[key].max())
        df["div_pay_amt_fillna"] = np.log(df["div_pay_amt_fillna"]+1)
        print(df.head())
        print(df.mean(axis=0))
        return df.values

    def read_data(self,simulate=False, normalize=False):
        start_time = time.time()
        if simulate:
            data = np.random.random((100000,self.ids + 2*self.state_dim+self.action_dim+1))
        else:
            data = self.read_from_csv("/home/xiao/Downloads/smalldata/pilushuju/model_input_sample_small_test.csv")
        # DATA IS A NUMPY ARRAY FORMAT
        end_time = time.time()
        print("Data loaded in {:.0f} seconds...".format(end_time-start_time))
        assert data.shape[1] == self.state_dim*2+self.action_dim+1 + 1 # added done here 
        return data


    def init_state_model(self, data):
        state_paras = []
        for i in range(self.state_dim):
            mean, std = norm.fit(data[self.ids :,self.ids + i])
            state_paras.append( (mean, std) )
        print("\t fitted initial state model successfully...")
        self.state_paras = state_paras


    def init_reward_model(self, data):
        X = self.data[:,self.ids:self.ids+self.state_dim+self.action_dim]
        y = self.data[:,-2]
        done = self.data[:,-1].astype('int')
        params = {"n_estimators":[30, 50, 75, 100, 125,150,175,200,225,250], "max_depth":[2,3,4]}
        reward_model = GridSearchCV(GradientBoostingRegressor(verbose=1,n_iter_no_change=20), params, n_jobs=-1).fit(X, y)
        print(reward_model)
        done_model = GradientBoostingClassifier().fit(X, done)
        print("GBDT Regression Prediction R2 = {:.4f}%".format(reward_model.score(X, y)*100))
        return reward_model, None, done_model


    def get_reward_from_action(self, state, action):
        '''action: one-hot list'''
        try:
            action = action.tolist()
        except Exception as e:
            pass
            
        if not isinstance(action, list): 
            actions = np.zeros(self.action_dim)
            actions[int(action)-1] = 1.
            action = actions
            
        X = np.append(state, action).reshape(1, -1).astype(float)
        
        reward = self.reward_model.predict(X)[0] 
        
        if np.random.randint(100)<10:
            reward = 0 
            
        if self.done: 
            reward = 0
        if not self.done: 
            if self.done_model.predict(X)[0]:
                self.done=True
            
        return reward

    def get_initial_state(self):
        state = []
        dict_state = {}
        for i in range(18):
            state_i = norm.rvs(self.state_paras[i][0], self.state_paras[i][1])
            state.append(state_i)
            dict_state[i] = float(state_i)
        # MANUALLY ADD THE NUMBER OF COUPONS CLIAMED AS 0 

        # 'buyer_dynamic_total_pay',
        state.append(0)
        dict_state[18]= 0

        # 'buyer_dynamic_sum_man_pert_0',
        state.append(0)
        dict_state[19]= 0

        # 'buyer_dynamic_avg_man_pert_0',
        state.append(0)
        dict_state[20]= 0

        # 'buyer_dynamic_num_transactions',
        state.append(0)
        dict_state[21]= 0

        # 'buyer_dynamic_num_0',
        state.append(0)
        dict_state[22]= 0

        # 'buyer_dynamic_num_receive',
        state.append(0)
        dict_state[23]= 0

        state = np.array(state)
        self.done=False
        return state, dict_state


    def post(self,url: str, content: Any) -> Any:
        last_error: Optional[urllib.error.HTTPError] = None
        for _retry in range(10):
            try:
                req = urllib.request.Request(url)
                req.add_header("Content-Type", "application/json; charset=utf-8")
                jsondataasbytes = json.dumps(content).encode("utf-8")  # needs to be bytes
                req.add_header("Content-Length", str(len(jsondataasbytes)))
                response = urllib.request.urlopen(req, jsondataasbytes)
                assert (
                    response.getcode() == 200
                ), "Error making request to ReAgent server: {} {}".format(
                    response.getcode(), response.read().decode("utf-8")
                )
                return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                print("Error: {} {}".format(e.getcode(), e.read().decode("utf-8")))
                last_error = e
                time.sleep(1)
        raise last_error


    def get_action_from_state(self, state, _post=True, _model=None):
        if state is not None:
            state_list = state.tolist()[0]
            d = {}
            for i in range(self.state_dim):
                d[i] = state_list[i]
        if _post:
            result = self.post(
                "http://localhost:3000/api/request",
                {
                    "plan_name": self.plan_name,
                    "context_features": d,  # "context_features": self.dict_state
                    "actions":{"names": self.ACTIONS} # added 20191115
                },
            )
            action_taken = str(result["actions"][0]["name"])
            action_prob_list = []
            action_probability = action_prob_list[int(action_taken)-1]
        else:
            result, action_taken, action_probability = self.predict_action_from_model(_model, request_id)
        
        action_list = torch.zeros([self.action_dim])
        action_list[int(action_taken)-1] = 1.
        action_taken = action_list
        return action_taken, action_probability


