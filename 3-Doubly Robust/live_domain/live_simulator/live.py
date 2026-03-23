import numpy as np
from scipy.integrate import odeint, ode
from src.config import live_config
from scipy.stats import norm, mode
from sklearn.linear_model import LinearRegression
import random
import pickle

class LiveTreatment(object):
    """
    Simulation of Live Treatment. The aim is to find an optimal targeting strategy.

    **STATE:** The state contains concentrations of 6 different cells:

    **N_STATE=100:**
    **N_ACTIONS:** 50

    """

    def __init__(self, scenario='SIMULATE',**kw):
        """
        Initialize the environment.
        """
        # Reorganized this function, Oct. 5, 2019
        # self.num_actions = 50  # do not need this, can be retrieved from config file
        self.state = None
        self.action = None
        self.state_dim = live_config.state_dim
        self.action_dim = live_config.action_size
        self.scenario = scenario
        self.reward_bound = 100 # randomly picked a number 100 to silence the warning.
        self.initialize()
        self.reset()
        
    # Rewrite this function on Oct. 5. 2019 
    def initialize(self):
        def _load_data(self):

            #### Why train data???
            self.raw_data = pickle.load(open("{}/{}_train.pickle".format(live_config.datadir, live_config.dataname), "rb")) # originally loading train.pickle, changed on Nov. 28
            print('filename',"{}/{}_train.pickle".format(live_config.datadir, live_config.dataname))
            self.data = self.raw_data.reset_index(drop=True).to_numpy()[:,3:] # drop two previous indexes # Originally [:,2:]
            assert self.data.shape[1] == 2*self.state_dim+self.action_dim+1, "shape of data is {}, while required is {}".format(self.data.shape[1], 2*self.state_dim+self.action_dim+1)

        def _fit_reward_model(self):
            X = self.data[:,:self.state_dim+self.action_dim]
            y = self.data[:,-1] # reward should be at the last index of each row 
            self.reward_model = LinearRegression().fit(X, y) # Simple linear Model may not work well, try others later
            print("\t fitted reward state model successfully...")
            
        def _fit_initial_state(self):
            # this implementation may be slow, find a package that support parallel fitting
            #print('initial_states original', self.raw_data.groupby("0").iloc['0270858417791a3e85b934a34954422a0'])
            self.initial_states = self.raw_data.groupby("0").first().reset_index(drop=True).to_numpy()[:,2:]

            #print('initial_states', self.raw_data.groupby("0").first().reset_index(drop=True).to_numpy()[0,2:])
            #print('initial_states:',self.initial_states.shape,self.initial_states)
            if self.scenario == "SIMULATOR":
                self.state_paras = []
                for i in range(self.action_dim+self.state_dim):
                    mean, std = norm.fit(self.initial_states[:,i])
                    self.state_paras.append( (mean, std) )
            print("\t fitted initial state model successfully...")
        
        def _fit_next_state_model(self):
            self.next_state_models = []
            for i in range(live_config.state_dim):
                X = self.data[:,:self.state_dim+self.action_dim]
                y = self.data[:,self.state_dim+self.action_dim+i]
                model = LinearRegression().fit(X, y) 
                self.next_state_models.append(model)
            print("\t fitted next state model successfully...")
            
        print("Initilizing Live Treatment ...")
        if self.scenario == 'SIMULATE':
            self.para_state = np.random.rand(self.state_dim)
            self.para_action = np.random.rand(self.action_dim)
            self.transition_matrix = np.random.dirichlet(np.ones(self.state_dim), size=self.action_dim * self.state_dim)
            self.transition_matrix = self.transition_matrix.reshape(self.state_dim, self.state_dim, self.action_dim)
        
        elif self.scenario == "SIMULATOR":
            _load_data(self)
            _fit_reward_model(self)
            _fit_next_state_model(self)
            _fit_initial_state(self)
            
        elif self.scenario == "USEBATCH":
            _load_data(self)
            _fit_initial_state(self)
            
        else:
            raise ValueError
        
    def reset(self, **kw): #changed on Oct 2, 2019
        """Reset the environment."""
        self.t = 0
        if self.scenario == 'SIMULATE':
            baseline_state = np.random.rand(self.state_dim) # in batch_data_processing.py # do not need to input tdimension here
            self.state = baseline_state
        elif self.scenario == 'SIMULATOR': # Added on Oct. 5, 2019 
            state = np.zeros(self.state_dim) 
            for i in range(self.state_dim):
                state[i] = norm.rvs(self.state_paras[i][0], self.state_paras[i][1])
            self.state = np.array(state)
        else: # 'USEBATCH' # Added on Oct. 5, 2019
            self.traj = random.choice(self.initial_states)
            self.state = self.traj[:self.state_dim]
            #print('reset state', self.initial_states[0][:self.state_dim])


    def observe(self):
        """Return current state."""
        return self.state

    def is_done(self, episode_length=200, **kw ):
        """Check if we've finished the episode."""
        #return True if self.t >= int(len(self.state)/(self.state_dim*2+self.action_dim+1)) else False
        return True if self.t >= episode_length else False
        
    def calc_reward(self, action=None, state=None, **kw ):
        """Calculate the reward for the specified transition."""
        # Think about deleting two lines below, Added Oct. 5, 2019
        if state is None:
            state = self.observe()
        # do not need to change class self state, commented out two lines below. Added Oct. 5, 2019 
        # self.state = state
        # self.action = action
        
        # the reward function penalizes treatment because of side-effects
        # reward = -0.1*V - 2e4*eps1**2 - 2e3*eps2**2 + 1e3*E # need to modify
        if self.scenario == 'SIMULATE':
            a = np.matmul(self.state, self.para_state)
            b = self.action * self.para_action[self.action]
            reward = a + b

        elif self.scenario == 'SIMULATOR':
            action_array = np.zeros(self.action_dim)
            action_array[action] = 1
            reward = self.reward_model.predict(np.append(state,action_array).reshape(1,-1)) 
        
        elif self.scenario == 'USEBATCH': 
            if len(self.traj)>(self.state_dim+self.action_dim+self.state_dim+1)*(self.t+1)-1: 
                reward = self.traj[(self.state_dim*2+self.action_dim+1)*(self.t+1)-1]
            else:
                reward = self.traj[-1]
            
        # Constrain reward to be within specified range
        if np.isnan(reward):
            reward = -self.reward_bound
        elif reward > self.reward_bound:
            reward = self.reward_bound
        elif reward < -self.reward_bound:
            reward = -self.reward_bound
            
        return reward

    def perform_action(self, state, action, **kw):
        """Perform the specifed action and upate the environment.
        Goal:
        given state, action, return next_state, reward

        Arguments:
        action -- action to be taken
        """
        self.t += 1
        self.action = action
        self.state = state
        self.state = self.next_state_function(action) # in batch_data_processing.py
        reward = self.calc_reward(state=state, action=action)
        return reward, self.observe()

    def next_state_function(self, action):
        '''
        batch data
        :param action and state: added parameter state to this function by Yan, 20190928
        :return: next state
        '''
        # if self.SIMULATE:
        if self.scenario == 'SIMULATE':
            dim = self.state.shape # this variable is not used below, Oct. 9, 2019 
            # next_state = np.zeros(dim)
            state_nextstate_matrix = self.transition_matrix[:, :, action]  # just a bad simulation
            next_state = np.matmul(self.state, state_nextstate_matrix)
            self.state = next_state
            return next_state
        
        elif self.scenario == 'SIMULATOR':
            # TODO: modelling next state given current state and action
            # Added on Oct. 9, 2019 
            next_state = []
            action_array = np.zeros(self.action_dim)
            action_array[action] = 1
            for i in range(len(self.next_state_models)):
                state_i = self.next_state_models[i].predict(np.append(self.state,action_array).reshape(1,-1))
                next_state.append(state_i[0])
            next_state = np.array(next_state)
            self.next_state = next_state
            return next_state 
        
        else: # 'USEBATCH'
            beg_idx = (self.state_dim*2+self.action_dim+1)*self.t
            end_idx = (self.state_dim*2+self.action_dim+1)*(self.t+1)
            next_state = self.traj[beg_idx:end_idx]
            return next_state 

        
        
# Test code, added Oct. 9, 2019 
if __name__ == "__main__":
    print("TESTING SIMULATOR...")
    lt = LiveTreatment(scenario="SIMULATOR") # test successfully, can initialize this class
    lt.reset() # test successfully, can be resetted 
    print( lt.is_done() ) # test successfully, can get status 
    print( lt.calc_reward(action=np.random.random(50)) ) # test successfully, can calculate reward 
    print( lt.next_state_function(action=np.random.random(50)) ) # test successfully, can go to next state 
    
    print("TESTING USEBATCH...")
    lt = LiveTreatment(scenario="USEBATCH") # test successfully, can initialize this class
    lt.reset() # test successfully, can be resetted 
    print( lt.is_done() ) # test successfully, can get status 
    action = pickle.load(open("./test_data_train.pickle", "rb")).to_numpy()[0][2:]
    action = action[100:150]
    print( lt.calc_reward(action=action) ) # test successfully, can calculate reward 
    print( lt.next_state_function(action=action) ) # test successfully, can go to next state
    
    
    
    