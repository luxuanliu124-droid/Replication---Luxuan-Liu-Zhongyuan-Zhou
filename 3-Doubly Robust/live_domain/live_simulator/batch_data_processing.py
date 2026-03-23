import numpy as np
from src.config import live_config
from live_domain.live_simulator.live import LiveTreatment
from live_domain.batch_policy_learning import BatchPolicyLearning
import pickle
import os
from scipy.stats import norm, mode
import pandas as pd

class BatchDataProcessing(object):
    
    def __init__(self, scenario='USEBATCH', state=None, action=None, preset_params=None,
                 state_dim=live_config.state_dim, action_dim=live_config.action_size, eval=True, upsample=False):
        self.scenario = scenario
        self.gamma = live_config.gamma
        self.episode_length = live_config.max_length
        self.state = state
        self.action = action
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.preset_params = preset_params
        self.eps = 0.0
        self.eval = eval
        self.upsample = upsample
        self.mode = "BCQ"
        self.datadir = live_config.datadir
        self.trajectories = []
        self.dataname_upsample = live_config.dataname_upsample 
        self.dataname = live_config.dataname 

        if self.scenario == 'SIMULATE':
            self.para_state = np.random.rand(self.state_dim)
            self.para_action = np.random.rand(self.action_dim)
            self.transition_matrix = np.random.dirichlet(np.ones(self.state_dim), size=self.action_dim * self.state_dim)
            self.transition_matrix = self.transition_matrix.reshape(self.state_dim, self.state_dim, self.action_dim)
        else:
            self._make_trajectories()
            self._fit_bpolicy()
        # changed the order of codes, Oct. 5, 2019
        # self.bpl = BatchPolicyLearning(scenario=self.scenario)
        self.bpl = None     # deprecated as we use USEBATCH 2020.01.03 Gaomin
        # self.task = LiveTreatment(self.scenario)

        
    # def _original_eval_split(self, USERID, TIMEID, train_pct=0.8):
    #     '''
    #     :function: train-test split to 80%-20%, called by _make_trajectories function
    #     :paras: userid index name and time id index name
    #     :return: None, save splitted data to data dir.
    #     '''
    #     df = pd.read_csv("{}/{}_full.csv".format(self.datadir, self.dataname), delimiter=",")
    #     df = df.reset_index().set_index(keys=[USERID, TIMEID]).sort_values(by=[USERID, TIMEID]).unstack(1)
    #     msk = np.random.rand(len(df)) <= train_pct
    #     data_original = df[msk].stack(1)
    #     data_eval = df[~msk].stack(1)
    #     pickle.dump(data_original, open("{}/{}_train.pickle".format(self.datadir, self.dataname),
    #                             "wb"), protocol=2)  # use this data to train model
    #     pickle.dump(data_eval, open("{}/{}_test.pickle".format(self.datadir, self.dataname), "wb"), protocol=2)

        
    def _make_trajectories(self):
        '''
         :called: This function is called when the class is initialized.
         :function: The main purpose of this function is to transform our output data to standardized trajectory set format
         
        '''
        # transform our data to trajectories format. May further require time variable to sorting, implemented
        USERID = "0"
        TIMEID = "1"
        #if not os.path.exists("{}/{}_train.pickle".format(self.datadir, self.dataname)):
        #    self._original_eval_split(USERID, TIMEID) # Changed on Nov 21, 2019 Don't use this b/c train test split is done by jupyter Use Zexi code
        if self.eval == True:  
            self.raw_data = pickle.load(open("{}/{}_train.pickle".format(self.datadir, self.dataname), "rb"))
            print('loaded train data')

        else: 
            if self.upsample == False:
                print("LOADING original data")
                self.raw_data = pickle.load(open("{}/{}_test.pickle".format(self.datadir, self.dataname), "rb"))
            else:
                print("LOADING upsample data")
                self.raw_data = pickle.load(open("{}/{}_test.pickle".format(self.datadir, self.dataname_upsample), "rb"))
            print('loaded test data')
        print(self.raw_data.shape)
        print('raw data has type {}'.format(self.raw_data.dtypes))
        self.data = self.raw_data.reset_index(drop=True).iloc[:,3:].to_numpy() #.astype(float)
        print(self.data.shape)
        print('self.data has type {}'.format(self.data.dtype))
            
        tmp = self.raw_data.reset_index().to_numpy()
        traj = []
        last = tmp[0][2]
        print("tmp.shape",tmp.shape)
        for i in range(tmp.shape[0]):
            obs = tmp[i]
            #changed on 12/03/2019
            #print('-----obs[2]=',obs[2])
            if obs[2] != last and len(traj)>0: #changed from obs[0] to obs[2] 12/03/2019
                last = obs[2]
                self.trajectories.append(traj)
                traj = []
            state = np.array(obs[4:4+self.state_dim]) 
            action = np.array(obs[4+self.state_dim:4+self.state_dim+self.action_dim]).argmax()
            next_state = np.array(obs[4+self.state_dim+self.action_dim:4+self.state_dim*2+self.action_dim])
            # changed on 12/03/2019
            reward = obs[-1]
            # removed as now: all log(reward+1) is done in data
            # reward=np.log(reward+1)
            traj.append( np.array([state, action, reward, next_state]) )   
        print("Made Trajactories successfully, with shape {}...".format(len(self.trajectories)))
        
        
    def _fit_bpolicy(self):
        actions = self.data[:,self.state_dim:self.state_dim+self.action_dim]
        print('actions has type {}'.format(actions.dtype))
        print("actions.shape=",actions.shape)
        print("actions=",actions[0,:])
        self.action_paras = []
        #for i in range(self.action_dim): # Changed Nov 21, 2019
        #    mean, std = norm.fit(actions[:,i]) # Changed Nov 21, 2019
        #    self.action_paras.append( (mean, std) ) # Changed Nov 21, 2019
        print('mean has type {}'.format(np.mean(actions, axis=0).dtype))
        self.action_paras.append(  np.mean(actions, axis = 0))     # Changed Nov 21, 2019
        print("Fitted behavior policy successfully... ")

        
    def run_episode(self, i_episode, eps=0.0, track = False):
        """Run an episode on the environment (and train Q function if modelfree)."""
        if self.scenario == 'USEBATCH': #changed on Oct 2, 2019
            #changed on 12/03/2019
            #print('-----generate_data/run_episode scenario=USEBATCH')
            ep_list = self.get_episode(i_episode) #changed on Oct 2, 2019
            return ep_list
        
        elif self.scenario == 'SIMULATE': #changed on Oct 2, 2019
            self.task.reset()
            state = self.task.observe()
            # task is done after max_task_examples timesteps or when the agent enters a terminal state
            ep_list = []
            action_list = []
            ep_reward = 0
            while not self.task.is_done(episode_length=self.episode_length):
                action_prob = np.random.dirichlet(np.ones(self.action_dim))
                action = np.random.multinomial(1,action_prob).argmax() # changed on Oct 1, 2019
                action_list.append(action)
                reward, next_state = self.task.perform_action(state, action)
                if track:
                    ep_list.append(np.array([state, action, reward, next_state]))
                state = next_state
                ep_reward += (reward*self.gamma**self.task.t)
            return ep_list
        
        else:  # 'SIMULATOR'
            #changed on 12/03/2019
            print('self.scenario =: ',self.scenario )
            print('-----generate_data/run_episode scenario=SIMULATOR')
            self.task.reset()
            state = self.task.observe()
            # task is done after max_task_examples timesteps or when the agent enters a terminal state
            ep_list = []
            action_list = []
            ep_reward = 0
            while not self.task.is_done(episode_length=self.episode_length):
                action_prob = self.bpl.policy(state, eps=self.eps)
                # action = np.random.multinomial(1, action_prob).argmax()  # changed on Oct 1, 2019, changed on Oct. 10,2019 
                action = action_prob.argmax()
                action_list.append(action)
                reward, next_state = self.task.perform_action(state, action)
                if track:
                    ep_list.append(np.array([state, action, reward, next_state]))
                state = next_state
                ep_reward += (reward * self.gamma ** self.task.t)
            return ep_list
        
        
    def get_episode(self, i_episode):
        # TODO: get ith episode 
        # Added Oct.6, 2019 
        traj = self.trajectories[i_episode]         
        # the format of  traj is [np.array(state1, action1, next_state1, reward1), np.array(state2, action2, next_state2, reward2)...]
        # should be the same as ep_list further transform this if needed.
        # print('i_episode', i_episode) 
        # print('get_episode traj=',traj)
        return traj 

    
    def bpolicy(self, state): #changed on Oct 2, 2019
        """Get the action under the behavioral policy for the given state.
        Args:
        state: The array of state features
        Returns:
        Behavioral Policy
        """
        if self.scenario == 'SIMULATE':
            print(np.random.dirichlet(np.ones(self.action_dim)).shape)
            return np.random.dirichlet(np.ones(self.action_dim))
        else:
            # USEBATCH and SIMULATOR,  # changed on Oct 2, 2019 # Added on Oct.5, 2019 
            # TODO Load and calcualte probability  # modified on Oct. 10, 2019
            action = np.zeros(self.action_dim)
            #for i in range(self.action_dim):
                #action[i] = norm.rvs(self.action_paras[i][0], self.action_paras[i][1])
            #print('debugging bpolicy !!!!!')
            #action = np.random.dirichlet(self.action_paras[0]) # Changed on Nov 21, 2019 
            action = self.action_paras[0] # Changed on Nov 22, 2019 
            #print('action shape', action.shape)
            #print('min action', action.min())
            #print('action param', self.action_paras[0])
            #print('action', action)
            #print('check whether action sum to 1', action.sum())
            return self.action_paras[0] # previously return action, changed on Nov.22
        
        
        
        
## test code, added Oct. 5, 2019 
if __name__== "__main__":
    bdp = BatchDataProcessing(scenario = "USEBATCH")
    print(len(bdp.trajectories))
#     print(bdp.trajectories[0])  
    print(bdp.run_episode(0))