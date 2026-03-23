import pandas as pd
import numpy as np
import pickle

# reproducible results
np.random.seed(2019)

file = '../../smalldata/model_input_sample_small_test_2020.csv'
data = pd.read_csv(file)

## 1. Split user and add percentage of pay

# add data pay or not to dataframe
df_add = data.copy()
df_add['pay_or_not'] = 1
df_add.loc[df_add.div_pay_amt_fillna==0,'pay_or_not'] = 0

# calculate buyers percentage of txn
user_count = df_add.groupby(['buyer_id']).count()[['receive_time_id']]
count_pay = df_add.groupby(['buyer_id']).sum()[['pay_or_not']]
data_user = pd.concat([user_count, count_pay],axis=1).rename(columns={'receive_time_id':'cnt_txn','pay_or_not':'cnt_pay'})
data_user['percent'] = data_user.cnt_pay.astype('int')/data_user.cnt_txn.astype('int')
data_user.reset_index(inplace=True)
# split all user into buy and never buy
data_user_never_buy = data_user[data_user['percent']==0]
data_user_buy = data_user[data_user['percent']!=0]

resample_buy_user_pool = list(np.random.choice(data_user_buy['buyer_id'], size=100, replace=False))
resample_never_buy_user_pool = list(np.random.choice(data_user_never_buy['buyer_id'], size=100, replace=False))
resample_user_pool = resample_buy_user_pool + resample_never_buy_user_pool

sample_data = data[data['buyer_id'].isin(resample_user_pool)]
print("# of sample data unique buyers:", len(sample_data['buyer_id'].unique()))
print("sample data shape:", sample_data.shape)
pickle.dump(sample_data, open("./orf_buyers_small_test.pickle", "wb"))
