class live_config():
    '''
    store hyperparameter for ORF
    '''

    # Data parameters
    num_state_var = 375
    num_action = 25

    # Model parameters
    num_trees = 200
    max_depth = 4
    min_leaf_size = 3
    lambda_reg = 0.01

    # Model save parameters
    model_save_path = './orf_saved.pkl'