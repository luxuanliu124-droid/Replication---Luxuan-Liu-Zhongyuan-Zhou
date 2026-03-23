
import pickle
import numpy as np
import time
import os
# this py file build lookup table according to orf train model
# Load trained model
# folder to save saved model
# folder = os.path.join('../', 'saved_model_191228')
import datetime


folder = os.path.join('./Benchmark_saved_model')
saved_orf_model = os.path.join(folder, 'orf_saved.pkl')
model = pickle.load(open(saved_orf_model, 'rb'))


# Load test data
test_data = pickle.load(open('./orf_buyers_small_test.pickle', 'rb')).values
test_data = test_data[:1000, 2:2+375]
print('test_data', test_data.shape)


#Estimate the treatment effect
start = time.time()
TE = model.const_marginal_effect(test_data)
print('TE', TE.shape)
end = time.time()
print('Estimation time elasped: ', end-start)
pickle.dump(TE, open(os.path.join(folder, 'orf_TE.pkl'), 'wb'))
# TE = pickle.load( open('../../live_rl-master_final/Benchmark_saved_model/orf_TE.pkl', 'rb'))

# # Make the action array
# argmax = np.argmax(TE, axis=1) + 1
# print('argmax', argmax.shape)
# output = np.zeros((test_data.shape[0],2), dtype=np.int)
# output[:,0] = argmax
# print('output', output.shape)
# mask = (np.sum(TE>0, axis=1) == 0).astype(int)
# print('mask', mask.shape)
#
# action_array = np.array([output[i, mask[i]] for i in range(test_data.shape[0])]) # batch_size * 1
# print('action_array', action_array.shape)
# action = np.eye(25)[action_array] # batch_size * 25
# print('action', action.shape)


# 20191227 make action proba
te_0 = np.zeros((TE.shape[0], 1))
te_25 = np.concatenate([te_0, TE], axis=1)
# overflow
te_25 = te_25 - te_25.max(axis=1, keepdims=True)
te_25 = np.exp(te_25)
row_sums = te_25.sum(axis=1)
action_25_prob = te_25/row_sums[:, np.newaxis]


dic = {}
for (state, action) in zip(test_data, action_25_prob):
	dic[tuple(state)] = action
print(dic)

pickle.dump(dic, open(os.path.join(folder, 'orf_lookup_table_proba.pkl'), 'wb'))
