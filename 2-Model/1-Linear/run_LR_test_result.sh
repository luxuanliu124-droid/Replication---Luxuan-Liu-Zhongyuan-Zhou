export CUDA_VISIBLE_DEVICES=""
python reformat_test_data_for_evaluation.py
python main_live_working_log_backup_debug_eval_ori.py --policy="LR" --train_batch_size=16 --train_num_batches=300
