from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
from gym.spaces import Box, Discrete, Dict
import time, logging

from a3gentnew.agents import agents
from a3gentnew.models import BatchDQNModel

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("tables", "", "tables names")
tf.flags.DEFINE_string("outputs", "odps://bp_ds_vs_dev/tables/sign_redpack_train_data_a3gent_r1_pred", "output tables names")
tf.flags.DEFINE_integer("task_index", 0, "index of worker in distributed job")
tf.flags.DEFINE_integer("num_actors", 1, "number of worker in distributed job") # request # of GPU
tf.flags.DEFINE_string("job_name", "worker", "job name: worker or ps")
tf.flags.DEFINE_string("oss_path_ckpt", "","")
tf.flags.DEFINE_string("oss_path_save", "","")

tf.flags.DEFINE_integer("batch_size", 256 , "") #2560 大一点 
tf.flags.DEFINE_integer("state_dim", 504,"")
tf.flags.DEFINE_integer("num_atoms", 1,"") # seems to be 1 in bcq, but 11 in dqn
tf.flags.DEFINE_integer("action_num", 17 ,"") # 16 in dqn, 17 in bcq

tf.flags.DEFINE_string("selected_cols","user_id,state,num_time_in_week","")
tf.flags.DEFINE_string("excluded_cols", "", "")

tf.flags.DEFINE_integer("epochs_to_end", 0,"") # 5/10 
tf.flags.DEFINE_integer("gamma", 95, "") #actual gamma is 0.95, input gamma/100, input gamma can only be integer

tf.flags.DEFINE_string("worker_hosts", "", "worker hosts")

# tf.flags.DEFINE_float("kl_weight", 0.001,"")
# tf.flags.DEFINE_string("target_dis","" ,"")

# set logger
logger = logging.getLogger("a3gentnew")
logger.setLevel(logging.DEBUG)

# create console handler
ch = logging.StreamHandler()
# create any formatter you like and add it to the handlers
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(ch)


MODEL_CONFIG = dict(
    # specific
    type="BCQ",
    n_step=1,
    dueling=False,
    double_q=True,

    # common
    gamma=float(FLAGS.gamma)/100.0, #0.95
    init_lr=1e-3,
    lr_strategy_spec={
        'type': 'exponential_decay',
        'decay_steps': 1000,
        'decay_rate': 0.9
    },
    clone_init_lr=1e-2,
    clone_lr_strategy_spec={
        'type': 'exponential_decay',
        'decay_steps': 1000,
        'decay_rate': 0.9
    },
    global_norm_clip=0.01)

AGENT_CONFIG = dict(
    type="BA",
    buffer_size=50000,
    learning_starts=1000,
    batch_size=FLAGS.batch_size,
    sync_target_frequency=1000,
)
np.random.seed(0)

class MyBCQModel(BatchDQNModel):
    def _generative_model(self, input_obs, scope="generative_model"):
        with tf.variable_scope(name_or_scope=scope):

            h1 = tf.layers.dense(
                input_obs,
                units=64,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(
                    mean=0.0, stddev=0.01, seed=0),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))
            h2 = tf.layers.dense(
                h1,
                units=64,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(
                    mean=0.0, stddev=0.01, seed=2),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))
            logits = tf.layers.dense(
                h2,
                units=FLAGS.action_num, #17
                activation=None,
                kernel_initializer=tf.random_normal_initializer(
                    mean=0.0, stddev=0.01, seed=5),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))

            return logits

    def _encode_obs(self, input_obs, scope="encode_obs"):
        with tf.variable_scope(name_or_scope=scope):
            h1 = tf.layers.dense(
                input_obs,
                units=64,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(
                    mean=0.0, stddev=0.01, seed=0),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))
            h2 = tf.layers.dense(
                h1,
                units=64,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(
                    mean=0.0, stddev=0.01, seed=1),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))
            h3 = tf.layers.dense(
                h2,
                units=32,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(
                    mean=0.0, stddev=0.01, seed=2),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))
            logits = tf.layers.dense(
                h3,
                units=FLAGS.action_num * MODEL_CONFIG["num_atoms"], #17
                activation=None,
                kernel_initializer=tf.random_normal_initializer(
                    mean=0.0, stddev=0.01, seed=0),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))
            return logits


class offline_env(object):

    def __init__(self, tables, task_index=0, num_actors=1, batch_size=FLAGS.batch_size):
        # create an offline_env to do fake interaction with agent
        self.num_epoch = 0
        self.num_record = 0
        self.task_index = task_index
        self.num_actors = num_actors

        # how many records to read from table at one time
        self.batch_size = batch_size

        # defined the shape of observation and action
        # we follow the definition of gym.spaces
        # `Box` for continue-space, `Discrete` for discrete-space and `Dict` for multiple input
        # actually low/high limitation will not be used by agent but required by gym.spaces
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(FLAGS.state_dim,))
        self.action_space = Discrete(n=FLAGS.action_num)

        self.tables = tables

        # more info about TableReader refer to: https://yuque.antfin-inc.com/pai-user/manual/py_table_reader
        self.table_reader = tf.python_io.TableReader(
            table=self.tables,
            selected_cols=FLAGS.selected_cols,
            slice_id=self.task_index,
            slice_count=self.num_actors
        )

    def parse_tuple_data(self, tuple_data):
        user_id, state, num_time_in_week = zip(*tuple_data)
        state_float = [[float(e) for e in elm.split(',')] for elm in state]
        num_time_in_week_int =[ int(elm)-1 if int(elm) >= 1 and int(elm) <=7 else 0 for elm in num_time_in_week]
        dict_data = dict(
            user_id = user_id,
            state_float=state_float,
            state_str =state,
            idx=num_time_in_week_int
        )
        return dict_data

    def reset(self):
        try:
            self.tuple_data = self.table_reader.read(self.batch_size)
        # pai-tf 1.8
        # except tf.errors.OutOfRangeError:
        # pai-tf 1.12
        except tf.python_io.OutOfRangeException:
            # reach the end of data
            return None
        self.num_record += len(self.tuple_data)
        dict_data = self.parse_tuple_data(self.tuple_data)

        return dict_data


def predict(worker_count, task_index):
    # create offline_env
    # create offline_env
    env = offline_env(FLAGS.tables, task_index=FLAGS.task_index, num_actors=worker_count,
                      batch_size=FLAGS.batch_size)
    print("outputs: {}".format(FLAGS.outputs))
    print("oss_path_ckpt: {}".format(FLAGS.oss_path_ckpt))
    writer = tf.python_io.TableWriter(FLAGS.outputs, slice_id=FLAGS.task_index)
    agent_class = agents[AGENT_CONFIG["type"]]
    # init agent
    agent = agent_class(
        env.observation_space,
        env.action_space,
        AGENT_CONFIG,
        MODEL_CONFIG,
        # set distributed_spec to {} in single-machine
        distributed_spec={},
        # set user-defined model
        custom_model=MyBCQModel,
        # set checkpoint path to save ckpt every 300 global_step by default
        # if set checkpoint_dir to none/empty ckpt will not be saved
        # variables will be restored before training if checkpoint_dir is not empty
        checkpoint_dir=FLAGS.oss_path_ckpt)

    total_cost = time.time()
    pred_count = 0

    # 0.1, 0.15, 0.21, 0.23, 0.28,   0.41, 0.5, 0.53, 0.66, 0.88,    1, 1.2, 1.3, 1.5, 1.8,   2 
    mask = np.array([[1, 1, 1, 1, 1,  1, 0, 1, 1, 1,  0, 0, 0, 0, 0,  0],
                     [1, 1, 1, 1, 1,  1, 0, 1, 1, 1,  0, 0, 0, 0, 0,  0], 
                     [1, 1, 1, 1, 1,  1, 0, 1, 1, 1,  0, 0, 0, 0, 0,  0], 
                     [0, 0, 0, 0, 0,  0, 1, 0, 0, 1,  1, 1, 0, 1, 0,  0], 
                     [1, 1, 1, 1, 1,  1, 0, 1, 1, 1,  0, 0, 0, 0, 0,  0], 
                     [1, 1, 1, 1, 1,  1, 0, 1, 1, 1,  0, 0, 0, 0, 0,  0], 
                     [0, 0, 0, 0, 0,  0, 0, 0, 0, 1,  0, 0, 1, 0, 1,  1] 
                        ]) # 7 * FLAGS.action_num array  ### Updated mask matrix on 02/08/2020
    # Change this mask matrix if restrict single action 

    while True:
        table_data = env.reset()
        if table_data:
            action, results, obs_embedding = agent.act(
                table_data["state_float"], deterministic=True)
            # action, results = agent.act(
            #     table_data["state_float"], deterministic=True)
            pred_count += len(action)

            obs_embedding = np.array(obs_embedding) # self.batch_size * FLAGS.action_num array 
            action_mask = mask[table_data["idx"]]  # self.batch_size * FLAGS.action_num array; idx must be within(0,7) 
            obs_embedding_masked = np.multiply(obs_embedding, action_mask) # element-wise product 
            action_masked = np.argmax(obs_embedding_masked, axis=1)
        else:
            break
        predictions = zip(table_data["user_id"], table_data["state_str"], action_masked)
        writer.write(predictions, indices=[0, 1, 2])


        if pred_count % (5*FLAGS.batch_size)==0:
            print("{}: {}".format(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()), pred_count))
    writer.close()
    print("Done.", "samples_count:", pred_count,
          "total_cost:", time.time() - total_cost)

def main():
    print("job name = %s" % FLAGS.job_name)
    print("task index = %d" % FLAGS.task_index)

    worker_spec = FLAGS.worker_hosts.split(",")
    cluster = tf.train.ClusterSpec({"worker": worker_spec})
    worker_count = len(worker_spec)

    predict(worker_count=worker_count, task_index=FLAGS.task_index)

if __name__ == "__main__":
    main()