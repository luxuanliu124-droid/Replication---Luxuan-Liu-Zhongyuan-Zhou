import json
import argparse
import os
import pickle
import math

parser = argparse.ArgumentParser()
parser.add_argument('--csv_file_name', 
                    type=str, 
                    default='/home/xiao/Downloads/smalldata/pilushuju/model_input_sample_small_train.csv')
parser.add_argument('--save_json_file_name', type=str, default='cartpole_discrete/data_123119.json')
parser.add_argument('--ds', type=str, default='2019-12-31')


args = parser.parse_args()

csv_file = args.csv_file_name
json_file = args.save_json_file_name

if os.path.exists(args.save_json_file_name):
    os.remove(args.save_json_file_name)
action_prob_from_data=[ ]   



# mean-std pair of each state variable 
mean_std = [       ]


f = open(csv_file)
f.readline()
count = 0
reward_sum = 0
action_d = {}
for row in f.readlines():
    count += 1
    if count % 1000 == 0: 
        print('total number of rows processed:',  count)
    d = {}
    # l = row.split(',')[:-1]
    l = row.split(',')
    l_action = [float(l[26 + i]) for i in range(25)]
    d["mdp_id"] = l[0]
    d["sequence_number"] = int(eval(l[1]))
    d["reward"] = round(math.log(float(l[-1]) + 1), 5)
    #d["reward"] = round(float(l[-1]), 5)
    # print(d["reward"])
    reward_sum += d["reward"]
    d["state_features"] = {}
    d["action"] = str(l_action.index(max(l_action)))
    if d["action"] in action_d:
        action_d[d["action"]] += 1
    else:
        action_d[d["action"]] = 1
    d["possible_actions"] = [str(j) for j in range(25)]
    #d["action_probability"] = max(l_action)
    #Xiao change 11/30/19
    d["action_probability"] = action_prob_from_data[l_action.index(max(l_action))]
    # print('---use action_prob from data',d["action_probability"])
    d["ds"] = args.ds # remember to change 'ds'

    for i in range(375):
        if l[i+2] == '':
            d['state_features'][str(i)] = 0
        else:
            #d['state_features'][str(i)] = round(float(l[i+2]),5) # changed to float
            try:
                d['state_features'][str(i)] = float(l[i+2])*mean_std[i][1]+mean_std[i][0]
                if i == 0 and d['state_features'][str(i)] > 100: 
                    # if the number of coupon received is greater than 100, replace it as 100. 
                    d['state_features'][str(i)] = 100
                if i == 1 and d['state_features'][str(i)] > 10000: 
                    # if the price of product is set above 10000 yuan, replace it as 10000 yuan 
                    d['state_features'][str(i)] = 10000
                if i == 2:
                    # if the number of product in inventory is above 100000, replace it as 100000; if is < 0, replace it as 0  
                    if d['state_features'][str(i)] < 0:
                        d['state_features'][str(i)] = 0
                    elif d['state_features'][str(i)] > 100000:
                        d['state_features'][str(i)] = 100000
                if i == 3 and d['state_features'][str(i)] > 1000: 
                    # if dynamic total pay is above 1000, replace it as 1000 
                    d['state_features'][str(i)] = 1000
                if i == 4 and d['state_features'][str(i)] > 10000:
                    # if prod stat avg reserve price > 10000, replace it as 10000 
                    d['state_features'][str(i)] = 10000
                if i == 5:
                    pass # do not change recency state var. 
                if i == 6 and d['state_features'][str(i)] > 10000: 
                    # if host live sta alipay item quantity > 10000, replace it as 10000 
                    d['state_features'][str(i)] = 10000 
                if i == 7 and d['state_features'][str(i)] > 200000: 
                    # if prod stat avg reserve price > 200000, replace it as 200000 
                    d['state_features'][str(i)] = 200000 
                if i == 8 and d['state_features'][str(i)] > 10000: 
                    # if host live stat alipay count > 10000, replace it as 10000 
                    d['state_features'][str(i)] = 10000 
                if i == 9 and d['state_features'][str(i)] > 1000000: 
                    # if host dyn total pay  > 1000000, replace it as 1000000 
                    d['state_features'][str(i)] = 1000000 
                if i == 10:
                    pass # do nothing about host live stat time lenth 
                if i == 11 and d['state_features'][str(i)] > 50000: 
                    # if number of unique buyers > 50000, replace it as 50000 
                    d['state_features'][str(i)] = 50000 
                if i == 12 and d['state_features'][str(i)] > 2000000: 
                    # if number of rate sum > 2000000, replace it as 2000000 
                    d['state_features'][str(i)] = 2000000 
                if i == 13:
                    pass # do not change service score 
                if i == 14: 
                    pass # do not change start time var. 
                if i == 15: 
                    pass # do not change fan value score 
                if i == 16: 
                    pass # do not change read time value 
                if i == 17 and d['state_features'][str(i)] > 20000:
                    # replace pv cart to 20000 if greater than 20000
                    d['state_features'][str(i)] = 20000 
                if i == 18 and d['state_features'][str(i)] > 10000:
                    # replace buyer dynamic total pay as 10000 if greater than 10000
                    d['state_features'][str(i)] = 10000 
                if i == 19 and d['state_features'][str(i)] > 3000:
                    # replace buyer tb mbr tq score as 3000 if greater than 4000
                    d['state_features'][str(i)] = 3000
                if i == 20 and d['state_features'][str(i)] > 40:
                     # replace man pert 0 as 40 if greater than 40
                    d['state_features'][str(i)] = 40
                if i == 21 and d['state_features'][str(i)] > 40:
                    # replace number of trasnactions to 40 if greater than 40
                    d['state_features'][str(i)] = 40
                if i == 22 and d['state_features'][str(i)] > 40:
                    # replace number of avg man to 40 if greater than 40
                    d['state_features'][str(i)] = 40
                if i ==23 and d['state_features'][str(i)] > 100:
                    # replace buyer dynamic number to 100 if greater than 100
                    d['state_features'][str(i)] = 100
                    
                d['state_features'][str(i)] = round(math.log(2 + d['state_features'][str(i)]), 5 )
            except Exception as e :
                d['state_features'][str(i)] = 0 
                print(i,float(l[i+2]), 2+float(l[i+2])*mean_std[i][1]+mean_std[i][0], e ) 
    with open(json_file, 'a') as js_f: # changed the output mode here 
        json.dump(d, js_f)
        js_f.write("\n")
f.close()
print("avg_reward:", reward_sum/count)
print('action:', action_d)
# to print a sample: $ cat data_112719.json | head -n1 | python -m json.tool









