import numpy as np

class SAMPLE(object):
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
        raise UnimplementedError()

    def sanity_check(self):
        '''
        This function do some necessary sanity checks to prevent unexpected errors. 
        paras: None 
        return: None 
        '''
        raise UnimplementedError()
        
    def select_action(self, state):
        '''
        # THIS FUNCTION MUST BE NAMED AS "select_action"
        The function returns an action given a state
        paras: state, in numpy array, in the simulated test data, should be a (100, ) numpy array
        return: action, in numpy array, in the simulated test data, should be a (50, ) numpy array
        Note that models use some package defined variables such as Tensors, need to transform those variables into plain numpy format. 
        '''
        raise UnimplementedError()

        
if __name__="__main__":
    sample = SAMPLE(modelpath=".")
    print(sample.select_action(np.random.random(100))
          
          