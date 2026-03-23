from main_live_working_log_backup_debug_eval_ori import args


class live_config():

    # domain parameters
    state_dim = 24
    action_size = 50 
    gamma = 0.99 # discount factor of reward
    max_length = 25 # changed on Nov 22, 2019, originally 200
###################################################################
# # parameter change for testing (wait time too long)
#     # model parameters
#     sample_num_traj = 10
#     train_num_traj = 15
#     dev_num_traj = 5
#     rep_hidden_dims = [64, 64] # The last layer is the representation dim
#     transition_hidden_dims = []
#     reward_hidden_dims = []
#     num_samples = 100 #used for evaluation, in compute_values, doubly robust, importance sampling

#     print_per_epi = 5
#     train_num_episodes = 5
#     train_num_batches = 5
#     train_batch_size = 4
#     test_batch_size = 5
#     train_traj_batch_size = 4
#     lr = 0.01
#     lr_decay = 0.9
#     alpha_rep = 0.1

#     # eval_num_traj = 1000
#     eval_num_rollout = 1
#     eval_pib_num_rollout = 5

#     N = 1
###################################################################
    # # model parameters
    # sample_num_traj = 200  # changed on 12/03/2019 old value 40 
    # train_num_traj = 200  # changed on 12/03/2019 old value 45 
    # dev_num_traj = 50 # changed on 12/03/2019 old value 5 
    # rep_hidden_dims = [8, 4] # The last layer is the representation dim
    # transition_hidden_dims = []
    # reward_hidden_dims = []
    # num_samples = 100 #used for evaluation, in compute_values, doubly robust, importance sampling
    
    # print_per_epi = 10
    # train_num_episodes = 100
    # train_num_batches = 100
    # train_batch_size = 40
    # test_batch_size = 5
    # train_traj_batch_size = 4
    # lr = 0.01
    # lr_decay = 0.9
    # alpha_rep = 0.1
    # eval_num_traj = 1000
    # eval_num_rollout = 1
    # eval_pib_num_rollout = 100

    # N = 1 # changed on 12/03/2019 old value is 10
###################################################################

###################################################################
    # model parameters
    sample_num_traj = 1  # =train_num_traj + dev_num_traj not used 
    train_num_traj = 5000  # used for training the mdpnet, max=10,000, bigger more accurate
    dev_num_traj = 4500 # used for testing the mdpnet
    rep_hidden_dims = [8, 4] # The last layer is the representation dim
    transition_hidden_dims = []
    reward_hidden_dims = []
    # changed: now use train_num_traj+dev_num_traj to eval
    num_samples = 50 #used for evaluation, in compute_values, doubly robust, importance sampling

    print_per_epi = 1
    train_num_episodes = train_num_traj    
    train_batch_size = int(20)
#     train_num_batches = int(400)
    train_num_batches = int(abs(train_num_traj/train_batch_size))+1  # too slow
    test_batch_size = 10 #not used
    train_traj_batch_size = 10 #not used
    lr = 0.01
    lr_decay = 0.9
    alpha_rep = 0.1
    eval_num_traj = 100 #not used
    eval_num_rollout = 1  #used in compute values for DM & DR
    eval_pib_num_rollout =100 

    N = 1 # changed on 12/03/2019 old value is 10
###################################################################    



    behavior_eps = 0.05
    standardize_rewards = True
    ins = 20 # may delete #changed on Oct 2, 2019

 #########################################################################
   
    # # added data dir and data name on Oct. 5 2019 
    # datadir = "."
    # dataname = "buyers_small_reformatted"
    # USERID = "0"
    # TIMEID = "1"
    # modelpath = "/home/xiao/Desktop/Jiawen/BCQ/"
    # modelname = "results/policy_checkpoints_100.0.pickle"

    # # Benchmark
    # # added data dir and data name on 2019.11.12
    # # modelpathBENCHMARK = "/Users/lx195541/Desktop/Xinyu_Zexi_Gaomin_1113sync/live_rl-master/Benchmark_saved_model/"
    # modelpathBENCHMARK = "./Benchmark_saved_model/"

    # modelnameGBDT = "gbdt_best_estimator_regression.sav"
    # modelnameORF = "orf_saved.pkl"
    # modelnameNN = "nn_model.pt"
    # modelnameNN50 = "nn_50_models"
    # modelnameLR = "LinearRegression1201.sav"

#########################################################################
##### Below is used by Xinyu  ########
    datadir = "."
    dataname = "buyers_small_reformatted"
    dataname_upsample = "7_{}_6_{}buyers_small_reformatted_upsample".format(args.num_samp_7,args.num_samp_6)
    print(dataname_upsample)
    USERID = "0"
    TIMEID = "1"
    modelpath = "/home/xiao/Desktop/Jiawen/BCQ/"
    modelname = "results/policy_checkpoints_100.0.pickle"

    # Benchmark
    # added data dir and data name on 2019.11.12
    # modelpathBENCHMARK = "/Users/lx195541/Desktop/Xinyu_Zexi_Gaomin_1113sync/live_rl-master/Benchmark_saved_model/"
    # policy = "LR" # Options: "LR", "GBDT", "ORF", "NN"
    modelpathBENCHMARK = "./Benchmark_saved_model/"

    modelnameGBDT50 = "gbdt_50_models"
    modelnameORF = "orf_saved.pkl"
    lookup_tableORF = 'orf_lookup_table_proba.pkl'
    modelnameNN50 = 'nn_50_models'
    modelnameLR = "LinearRegression.sav"