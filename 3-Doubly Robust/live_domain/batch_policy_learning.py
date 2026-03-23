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

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class BCQ_model():
    def __init__(self, modelpath):
        self.load_model(modelpath)
        self.sanity_check()
        
    def sanity_check(self):
        pass 
    
    def load_model(self, modelpath):
        print("Loading model from {}".format(modelpath)) 
        self.model = pickle.load(open(modelpath, "rb"))
        
    def select_action(self, state):
        assert state.shape[0]==100
        return self.model.select_action(state) 
        # batch ? 
        # this model.select_action is customizable, depending on your training algo. 

class LRBaseline(object):
    '''
    ########################################
    ##### POLICY LEARNING INPUT SAMPLE #####
    ########################################
    This class should be called as interface to load learned policies.
    This class mainly have three functions but may subject to change to further accomodate our data and codes.
    Scikit learn version: '0.20.2'
    '''

    def __init__(self, modelpath):
        self.num_state_var = live_config.state_dim
        self.num_action = live_config.action_size
        self.load_policy(modelpath)
        self.sanity_check()



    def load_policy(self, modelpath):
        '''
        The output file should be loaded in the function and stored in memory for further use.
        paras: modelpath
        return: None
        '''

        self.model = pickle.load(open(modelpath, 'rb'))
        print('LR Model successfully loaded from {}'.format(modelpath))
        print('\n Hyperparameter:{}\n num_state_var:{}\n num_action:{}\n'.format('', # Xinyu changed on Nov.30, originally model_dict['hyperparameter']
                                                                               self.num_state_var,
                                                                                   self.num_action))

    def sanity_check(self):
        '''
        This function do some necessary sanity checks to prevent unexpected errors.
        paras: None
        return: None
        '''
        pass

    def select_action(self, states):
        '''
        # changed 12/26/2019 instead of return the best action, now return action prob
        # THIS FUNCTION MUST BE NAMED AS "select_action" 
        The function returns an action given a state [one state]
        paras: state, in numpy array, in the simulated test data, should be a (100, ) numpy array
        return: action, in numpy array, in the simulated test data, should be a (50, ) numpy array
        Note that models use some package defined variables such as Tensors, need to transform those variables into plain numpy format.
        '''
        if not isinstance(states, np.ndarray):
            raise TypeError('Expected input to be a numpy array but got {}'.format(type(states)))
        # if states is not a 2d array reshape it
        try:
            print("num_state_var:", states.shape[1])
        except:
            states = states.reshape((1, states.shape[0]))

        if not states.shape[1] == self.num_state_var:
            raise ValueError('states should in shape ( ,{})'.format(self.num_state_var))

        model = self.model

        resArray = np.zeros((states.shape[0], self.num_action))
        for i in range(self.num_action):
            actions_temp = np.zeros((states.shape[0], self.num_action))
            actions_temp[:, i] = 1
            X = np.concatenate([states, actions_temp], axis=1)
            pred_rewards = model.predict(X)
            resArray[:, i] = pred_rewards
        # normalize reward in each row as prob
        resArray = np.exp(resArray)
        row_sums = resArray.sum(axis=1)
        res = resArray / row_sums[:, np.newaxis]
        res = res.reshape(-1)
        if np.isnan(res).any():
            print(resArray)
            print(row_sums)

        return res


class GBDTBaseline50(object):
    '''
    ########################################
    ##### POLICY LEARNING INPUT SAMPLE #####
    ########################################
    This class should be called as interface to load learned policies.
    This class mainly have three functions but may subject to change to further accomodate our data and codes.
    Scikit learn version: '0.20.2'
    '''

    def __init__(self, modelpath):
        self.load_policy(modelpath)
        self.sanity_check()




    def load_policy(self, modelpath):
        '''
        Load model list of 50 models in the list
        paras: modelpath
        return: None
        '''
        self.model_list = []
        for file_name in ['gbdt_best_estimator_model{}_.sav'.format(i) for i in range(50)]:
            model_dict = pickle.load(open(os.path.join(modelpath, file_name), 'rb'))
            self.model_list.append(model_dict['model'])
            print('Load GBDT model ',file_name)
        self.num_state_var = model_dict['num_state_var']
        self.num_action = model_dict['num_action']
        print('GBDT Model successfully loaded from {}'.format(modelpath))
        

    def sanity_check(self):
        '''
        This function do some necessary sanity checks to prevent unexpected errors.
        paras: None
        return: None
        '''
        pass

    def select_action(self, states):
        '''
        # THIS FUNCTION MUST BE NAMED AS "select_action"
        The function returns an action given a state [one state]
        paras: state, in numpy array, in the simulated test data, should be a (100, ) numpy array
        return: action, in numpy array, in the simulated test data, should be a (50, ) numpy array
        Note that models use some package defined variables such as Tensors, need to transform those variables into plain numpy format.
        '''
        if not isinstance(states, np.ndarray):
            raise TypeError('Expected input to be a numpy array but got {}'.format(type(states)))
        # if states is not a 2d array reshape it
        try:
            print("num_state_var:", states.shape[1])
        except:
            states = states.reshape((1, states.shape[0]))

        if not states.shape[1] == self.num_state_var:
            raise ValueError('states should in shape ( ,{})'.format(self.num_state_var))

        resArray = np.zeros((states.shape[0], self.num_action))
        for i in range(len(self.model_list)):
            model = self.model_list[i]
            X = states
            pred_rewards = model.predict(X)
            resArray[:, i] = pred_rewards


        # normalize reward in each row as prob
        # solve the overflow problem
        resArray = resArray - resArray.max(axis=1, keepdims=True)
        # exp overflow
        resArray = np.exp(resArray)
        row_sums = resArray.sum(axis=1)

        res = resArray / row_sums[:, np.newaxis]
        res = res.reshape(-1)

        return res


class ORF_policy_learning(object):
    '''
    This class should be called as interface to load learned policies.
    This class mainly have three functions but may subject to change to further accomodate our data and codes.
    '''

    def __init__(self, modelpath):
        self.load_policy(modelpath)
        self.sanity_check()


    def load_policy(self, modelpath):
        '''
        The output file should be loaded in the function and stored in memory for further use.
        paras: modelpath
        return: None
        '''

        # self.model = pickle.load(open(modelpath, 'rb'))
        self.lookup_table = pickle.load(open(modelpath, 'rb'))

        self.lookup_states = self.lookup_table.keys()

        self.num_state_var = live_config.state_dim


        print('ORF Model successfully loaded from {}'.format(modelpath))


    def sanity_check(self):
        '''
        This function do some necessary sanity checks to prevent unexpected errors.
        paras: None
        return: None
        '''
        pass

    def select_action(self, states):
        '''
        # THIS FUNCTION MUST BE NAMED AS "select_action"
        The function returns an action given a state [one state]
        paras: state, in numpy array, in the simulated test data, should be a (num_state_var, ) numpy array
        return: action, in numpy array, in the simulated test data, should be a (50, ) numpy array
        '''
        if not isinstance(states, np.ndarray):
            raise TypeError('Expected input to be a numpy array but got {}'.format(type(states)))
        # if states is not a 2d array reshape it
        try:
            print("num_state_var:", states.shape[1])
        except:
            states = states.reshape((1, states.shape[0]))

        if not states.shape[1] == self.num_state_var:
            raise ValueError('states should in shape ( ,{})'.format(self.num_state_var))


        input_state = tuple(states[0])
        lookup_states = list(self.lookup_states)
        # diff of list of tuples and tuple
        diff = np.array([tuple(map(lambda i, j: i - j, e, input_state))  for e in lookup_states])
        squared_diff = np.square(diff)
        mean_squared_diff = np.mean(squared_diff, axis=1)

        ind = np.argmin(mean_squared_diff)
        closest_state = lookup_states[ind]
        resArray = self.lookup_table[closest_state]

        # Note resArray here is in a 1d manner
        # normalize reward in each row as prob
        # solve the overflow problem
        resArray = resArray - resArray.max()
        resArray = np.exp(resArray)
        row_sums = resArray.sum()
        res = resArray / row_sums
        res = res.reshape(-1)

        return res

'''
The following block is the implementation of neural net model based upon 50 sub-models
'''
class BaselineNeuralNet50Models(object):
    '''
    ########################################
    ##### POLICY LEARNING INPUT SAMPLE #####
    ########################################
    This class should be called as interface to load learned policies.
    This class mainly have three functions but may subject to change to further accomodate our data and codes.
    '''
    
    def __init__(self, modelpath):
        self.load_policy(modelpath)
        self.sanity_check()
        
        
    def load_policy(self, modelpath):
        '''
        The output file should be loaded in the function and stored in memory for further use. 
        paras: None 
        return: None 
        '''
        self.instance = BaselineNNInstance50Models(24, 50) # num_states and num_actions are hard-coded
        self.instance.loadWeights(modelpath)


    def sanity_check(self):
        '''
        This function do some necessary sanity checks to prevent unexpected errors. 
        paras: None 
        return: None 
        '''
        pass
        
    def select_action(self, state):
        '''
        # THIS FUNCTION MUST BE NAMED AS "select_action"
        The function returns an action given a state
        paras: state, in numpy array, in the simulated test data, should be a (100, ) numpy array
        return: action, in numpy array, in the simulated test data, should be a (50, ) numpy array
        Note that models use some package defined variables such as Tensors, need to transform those variables into plain numpy format. 
        '''
        
        res = self.instance.predictOptimalAction(state)

        return res


################## Underlying caller of the neural net 50 models ##################

class BaselineNNInstance50Models(object):

    def __init__(self, num_state_var, num_action, dropout_rate=0.2):
        '''
        Parameters
        ----------
        num_state_var: int
            Number of state variables
        num_action: int
            Number of action types
        dropout_rate: float, default 0.2
            Dropout rate used in FC layers

        '''
        # input_dim = num_state_var + num_action
        self.model_list = [BaselineFCStates(num_state_var, dropout_rate) for _ in range(num_action)]
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        # self.model.to(self.device)
        self.num_state_var = num_state_var
        self.num_action = num_action

    def loadWeights(self, model_path):
        '''
        Load saved weights into model
        '''
        for i, m in enumerate(self.model_list):
            m.load_state_dict(torch.load('{}/nn_action_{}.pt'.format(model_path, i)))
        print('Weights loaded from {}'.format(model_path))

    def saveWeights(self, model_path):
        '''
        Save weights
        '''
        for i, m in enumerate(self.model_list):
            torch.save(m.state_dict(), '{}/nn_action_{}.pt'.format(model_path, i))
        print('Weights saved to {}'.format(model_path))

    def trainAll(self,
                train_data,
                states_val=None,
                rewards_val=None,
                num_epochs=10,
                batch_size=32,
                lr=0.005,
                eval_on_valid=False):
        '''
        Train all models in self.model_list
        Takes in all train data, group by action_type, train separately

        '''
        
        # Split training data into 50 actions
        action_matrix = train_data.iloc[:, 2 + self.num_state_var: 2 + self.num_state_var + self.num_action].values
        print('action_matrix has shape {}'.format(action_matrix.shape))
        train_data['action'] = action_matrix.argmax(axis=1)
        
        train_grp = train_data.groupby('action')

        for action_type, df in train_grp:
            print('*' * 60)
            print('*' * 20, 'TRAINING MODEL FOR ACTION_TYPE {}'.format(action_type), '*' * 20)
            print('*' * 60)
            print()

            df = df.drop(columns=['action'])
            print('df has shape {}'.format(df.shape))

            # Get state and reward vars
            states_train, rewards_train = df.iloc[:, 2: 2 + self.num_state_var].values, df.iloc[:, -1].values

            # Train a model
            self.train(action_type,
                    states_train,
                    rewards_train,
                    states_val,
                    rewards_val,
                    num_epochs,
                    batch_size,
                    lr,
                    eval_on_valid)

    def train(self,
            action_type,
            states_train,
            rewards_train,
            states_val=None,
            rewards_val=None,
            num_epochs=10,
            batch_size=32,
            lr=0.005,
            eval_on_valid=False):
        '''
        Train a single model in self.model_list using training data associated with the specified action_type
        
        '''

        # Turn training data into DataLoaders
        dataset_train = TensorDataset(torch.Tensor(states_train),
                                    torch.Tensor(rewards_train).unsqueeze(dim=1)) # Convert rewards_train from 1-d to 2-d

        dataloader_train = DataLoader(dataset_train, batch_size=batch_size)

        # Turn validation data into Dataloaders
        if eval_on_valid:
            dataset_val = TensorDataset(torch.Tensor(states_val),
                                    torch.Tensor(rewards_val).unsqueeze()) # Convert rewards_train from 1-d to 2-d

            dataloader_val = DataLoader(dataset_val, batch_size=batch_size)


        # Let model reference self.model 
        model = self.model_list[action_type]

        # Criterion
        criterion = nn.MSELoss()
        # Optimizer
        optimizer = optim.Adam([param for param in model.parameters() if param.requires_grad], lr=lr)

        optimizer.zero_grad()

        for epoch in range(num_epochs):
            # Train
            model.train()
            print('=' * 30, 'Epoch {}'.format(epoch + 1), '=' * 30)
            train_loss = 0
            for i, (s, t) in enumerate(dataloader_train):
                optimizer.zero_grad()
                output = model(s)
                loss = criterion(output, t)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
                
                # print average training loss every 50 steps
                if i % 50 == 0 or i == len(dataloader_train) - 1:
                    print('Step {}/{}, Avg train loss {}'.format(i+1, len(dataloader_train), train_loss / (i+1)))
            
            # Evaluate on valid set
            if eval_on_valid:
                with torch.no_grad():
                    model.eval()
                    val_loss = 0
                    for s, t in dataloader_val:
                        output = model(s)
                        loss = criterion(output, t)
                        val_loss += loss.item()
                    
                    # print average validation loss
                    print('Avg val loss {}'.format(val_loss / len(dataloader_val)))

            print()


    def predictOptimalAction(self, states):
        '''
        Predict the action that the model believes will induce the maximum reward
        
        '''
        if not isinstance(states, np.ndarray):
            raise TypeError('Expected input to be a numpy array but got {}'.format(type(states)))
        
        # print('Predicting...')
        # print('Type of states is {}'.format(type(states)))
        # print('Shape of states is {}'.format(states.shape))
        # print('dtype of states is {}'.format(states.dtype))
        # print('State is')
        # print(states)
        # assert states.shape[1] == self.num_state_var

        states = states.astype(float)
        # print('dtype of states after conversion is {}'.format(states.dtype))
        # Convert state vars into Tensors
        states = torch.Tensor(states)
        

        # Initialize an array to record predicted rewards
        # resArray = np.zeros((states.shape[0], self.num_action))
        resArray = np.zeros(self.num_action)

        # For each of the 50 models, predict rewards and save them in resArray
        for action_type in range(self.num_action):
            # print('-----action_type=',action_type)
            model = self.model_list[action_type]
            model.eval()
        
            with torch.no_grad():
                pred_rewards = model(states)
                # resArray[:, action_type] = pred_rewards.detach().numpy().ravel()
                resArray[action_type] = pred_rewards.detach().item()

        # Note resArray here is in a 1d manner
        # normalize reward in each row as prob
        # solve the overflow problem
        resArray = resArray - resArray.max()
        resArray = np.exp(resArray)
        row_sums = resArray.sum()
        res = resArray / row_sums
        res = res.reshape(-1)

        return res
        
        # # Reformat result
        # # resArrayArgMax = resArray.argmax(axis=1)
        # resArrayArgMax = resArray.argmax()
        
        # res = np.zeros(self.num_action)
        # res[resArrayArgMax] = 1
        # # res = np.eye(self.num_action)[resArrayArgMax]
        # # print('---------res.shape=',res.shape)
        # return res 


################## Underlying Neural Net ##################

class BaselineFCStates(nn.Module):

    def __init__(self, input_dim, dropout_rate=0.2):
        super(BaselineFCStates, self).__init__()
        # Network architecture
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, states):
        out = self.dropout(self.fc1(states))
        out = F.relu(out)
        out = self.dropout(self.fc2(out))
        out = F.relu(out)
        out = self.fc3(out)
        return out


    
class BatchPolicyLearning(object):
    '''
    in this class, we load trained policy function
    '''
    def __init__(self, scenario='SIMULATE', ModelPolicy ='LR'):
        gc.disable()
        self.state = None
        self.scenario = scenario
        self.ModelPolicy = ModelPolicy

        if self.scenario == "SIMULATOR" or self.scenario == "USEBATCH":# "SIMULATOR",'USEBATCH'
            if not self.ModelPolicy in ["LR", "GBDT", "ORF", "NN"]:
                raise ValueError('Expected policy to be one of ["LR", "GBDT", "ORF", "NN"] but got {}'.format(self.ModelPolicy))
            
            if self.ModelPolicy == 'LR':
                #Linear Regression
                print('POLICY MODEL: Linear Regression')
                sys.path.append(live_config.modelpathBENCHMARK)
                self.modelpath = live_config.modelpathBENCHMARK+live_config.modelnameLR
                self.model = LRBaseline(self.modelpath)

            if self.ModelPolicy == 'GBDT':
                # GBDT 50
                print('POLICY MODEL: GBDT 50')
                sys.path.append(live_config.modelpathBENCHMARK)
                self.modelpath = live_config.modelpathBENCHMARK+live_config.modelnameGBDT50
                self.model = GBDTBaseline50(self.modelpath)

            if self.ModelPolicy == 'ORF':
                # #ORF
                print('POLICY MODEL: ORF')
                sys.path.append(live_config.modelpathBENCHMARK)
                self.lookup_table_path = live_config.modelpathBENCHMARK+live_config.lookup_tableORF 
                self.model = ORF_policy_learning(self.lookup_table_path)

            if self.ModelPolicy == 'NN':
                # NN 50
                print('POLICY MODEL: NN 50')
                sys.path.append(live_config.modelpathBENCHMARK)
                self.modelpath = live_config.modelpathBENCHMARK+live_config.modelnameNN50
                self.model = BaselineNeuralNet50Models(self.modelpath)            
            
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

