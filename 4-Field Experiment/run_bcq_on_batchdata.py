# Copyright (c) 2019 Alibaba Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import time
import numpy as np
from gym.spaces import Box, Discrete

from a3gentnew.agents import agents
from a3gentnew.models import BatchDQNModel
from a3gentnew.utils.window_stat import WindowStat
from a3gentnew.utils.inverse_propensity_score import ips_eval

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("tables", "", "tables names")
tf.flags.DEFINE_string("oss_path_ckpt", "oss://netspace/work/gongxiangdm/qinjian/tmallchaoshi/BCQ_V1/model/ckpt/","")
tf.flags.DEFINE_string("oss_path_save", "oss://netspace/work/gongxiangdm/qinjian/tmallchaoshi/BCQ_V1/model/save/","")
tf.flags.DEFINE_integer("batch_size", 256 , "")
tf.flags.DEFINE_integer("num_atoms", 1,"") # seems to be 1 in bcq, but 11 in dqn
tf.flags.DEFINE_integer("action_num", 17 ,"") # 16 in dqn, 17 in bcq
tf.flags.DEFINE_integer("state_dim", 504,"")
tf.flags.DEFINE_string("selected_cols","user_id, time_id, state, action, reward, terminal, next_state","") 

tf.flags.DEFINE_integer("epochs_to_end", 10,"")
tf.flags.DEFINE_integer("max_globel_steps_to_end", 10000,"")
tf.flags.DEFINE_integer("gamma", 95, "") #actual gamma is 0.95, input gamma/100, input gamma can only be integer


MODEL_CONFIG = dict(
    # specific
    type="BCQ",
    n_step=1,
    dueling=False,
    double_q=True,

    num_atoms=FLAGS.num_atoms,

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

    def __init__(self, tables, batch_size=128, n_step=1):
        # create an offline_env to do fake interaction with agent
        self.num_epoch = 0
        self.num_record = 0

        # how many records to read from table at one time
        self.batch_size = batch_size
        # number of step to reserved for n-step dqn
        self.n_step = n_step

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
            excluded_cols="",
            slice_id=0,
            slice_count=1
        )

    def parse_tuple_data(self, tuple_data, with_appendix=False):
        #assert len(tuple_data) == self.batch_size + self.n_step - 1

        if with_appendix:
            user_id, time_id, state, action, reward, terminal, next_state = zip(*tuple_data)
            time_id = [int(e) for e in time_id]
        else:
            user_id = time_id = None
            state, action, reward, terminal, next_state = zip(*tuple_data)

        state = [[float(e) for e in elm.split(',')] for elm in state]
        next_state = [[float(e) for e in elm.split(',')] for elm in next_state]
        terminal_ = []
        for e in terminal:
            if e.lower() == "false":
                terminal_.append(False)
            else:
                terminal_.append(True)

        reward = [float(e) for e in reward]
        action = [int(e) for e in action]

        dict_data = dict(
            user_id=user_id,
            time_id=time_id,
            obs=state,
            actions=action,
            rewards=reward,
            dones=terminal_,
            next_obs=next_state)

        return dict_data

    def reset(self):
        read_batch_size = self.batch_size + self.n_step - 1
        try:
            self.tuple_data = self.table_reader.read(read_batch_size)
            if len(self.tuple_data) < read_batch_size:
                # reach the end of data
                self.num_epoch += 1
                self.table_reader.close()
                self.table_reader = tf.python_io.TableReader(
                    table=self.tables,
                    selected_cols=FLAGS.selected_cols,
                    excluded_cols="",
                    slice_id=0,
                    slice_count=1
                )
                self.tuple_data.extend(self.table_reader.read(read_batch_size - len(self.tuple_data)))
        # pai-tf 1.8
        # except tf.errors.OutOfRangeError:
        # pai-tf 1.12
        except tf.python_io.OutOfRangeException:
            # reach the end of data
            self.num_epoch += 1
            self.table_reader.close()
            self.table_reader = tf.python_io.TableReader(
                table=self.tables,
                selected_cols=FLAGS.selected_cols,
                excluded_cols="",
                slice_id=0,
                slice_count=1
            )
            self.tuple_data = self.table_reader.read(read_batch_size)
        self.num_record += self.batch_size

        dict_data = self.parse_tuple_data(self.tuple_data, with_appendix=True)

        return dict_data

def main():
    # create offline_env
    env = offline_env(FLAGS.tables, batch_size=128, n_step=MODEL_CONFIG.get("n_step", 1))
    eval_env = offline_env(FLAGS.tables, batch_size=128, n_step=MODEL_CONFIG.get("n_step", 1))

    agent_class = agents[AGENT_CONFIG["type"]]
    agent = agent_class(
        env.observation_space,
        env.action_space,
        AGENT_CONFIG,
        MODEL_CONFIG,
        distributed_spec={},
        export_dir="",
        checkpoint_dir=FLAGS.oss_path_ckpt,
        custom_model=MyBCQModel)

    clone_loss_window = WindowStat("clone_loss", 50)
    clone_reg_loss_window = WindowStat("clone_reg_loss", 50)
    loss_window = WindowStat("loss", 50)

    total_cost = time.time()
    clone_learn_count = 0

    # first, train a generative model by behavior clone

    for i in range(500):
        table_data = env.reset()
        clone_loss, clone_reg_loss = agent.behavior_learn(batch_data=table_data)
        clone_learn_count += 1
        clone_loss_window.push(clone_loss)
        clone_reg_loss_window.push(clone_reg_loss)
        if i % 50 == 0:
            print(clone_loss_window)
            print(clone_reg_loss_window)

    # test
    all_clone_act, gd_act = [], []
    for i in range(100):
        table_data = env.reset()
        clone_act = agent.behavior_act(table_data["obs"])
        all_clone_act.extend(np.argsort(-1.0 * clone_act, axis=1).tolist())
        gd_act.extend(table_data["actions"])
    acc1 = np.sum(np.array(all_clone_act)[:, 0] == np.array(gd_act))*1.0/len(gd_act)
    acc5 = np.sum(np.array(all_clone_act)[:, :5] == np.tile(np.expand_dims(np.array(gd_act), -1),[1,5]))*1.0/len(gd_act)
    print("acc @top1:", acc1, "acc @top5:", acc5)

    # second, train bcq
    agent.reset_global_step()

    epochs_to_end = FLAGS.epochs_to_end #10
    max_globel_steps_to_end = FLAGS.max_globel_steps_to_end #10000

    learn_count = 0
    env.num_epoch = 0
    while env.num_epoch < epochs_to_end and learn_count < max_globel_steps_to_end:
        table_data = env.reset()

        # store raw data in replay buffer
        agent.send_experience(
            obs=table_data["obs"],
            actions=table_data["actions"],
            rewards=table_data["rewards"],
            dones=table_data["dones"],
            next_obs=table_data["next_obs"])

        # sample from replay buffer
        # the size of sampled data is equal to `AGENT_CONFIG["batch_size"]`
        batch_data = agent.receive_experience()
        # update the model
        res = agent.learn(batch_data)
        # record the loss
        loss_window.push(res["loss"])
        learn_count += 1

        if AGENT_CONFIG.get("prioritized_replay", False):
            # update priorities
            agent.update_priorities(
                indexes=batch_data["indexes"],
                td_error=res["td_error"])

        if learn_count % 50 == 0:
            print("learn_count:", learn_count, "num_record:", env.num_record)
            print(loss_window)

            # evaluation
            raw_records = {}
            eval_num = 50
            for _ in range(eval_num):
                batch_data = eval_env.reset()
                for idx, key in enumerate(batch_data["user_id"]):
                    if key not in raw_records:
                        raw_records[key] = {}
                    step_id = batch_data["time_id"][idx]
                    ob = batch_data["obs"][idx]
                    action = batch_data["actions"][idx]
                    reward = batch_data["rewards"][idx]
                    next_ob = batch_data["next_obs"][idx]
                    terminal = batch_data["dones"][idx]

                    raw_records[key][step_id] = (ob, action, reward, terminal, next_ob,)

            batch_weights = []
            batch_rewards = []

            for key in raw_records:
                traj_obs = []
                traj_actions = []
                traj_rewards = []
                traj_terminals = []
                traj_next_obs = []
                for step_id in sorted(raw_records[key]):
                    traj_obs.append(raw_records[key][step_id][0])
                    traj_actions.append(raw_records[key][step_id][1])
                    traj_rewards.append(raw_records[key][step_id][2])
                    traj_terminals.append(raw_records[key][step_id][3])
                    traj_next_obs.append(raw_records[key][step_id][4])

                traj_batch_data = dict(
                    obs=traj_obs,
                    actions=traj_actions,
                    dones=traj_terminals,
                    rewards=traj_rewards,
                    next_obs=traj_next_obs
                )

                importance_ratio = agent.importance_ratio(traj_batch_data)
                batch_weights.append(importance_ratio)
                batch_rewards.append(traj_batch_data["rewards"])
            ips, ips_sw, wips, wips_sw, wips_sw_mean = ips_eval(
                batch_weights=batch_weights, batch_rewards=batch_rewards, gamma=MODEL_CONFIG.get("gamma", 0.95))

            agent.add_extra_summary({agent.model.ips_score_op:ips,
                                     agent.model.ips_score_stepwise_op:ips_sw,
                                     agent.model.wnorm_ips_score_op:wips,
                                     agent.model.wnorm_ips_score_stepwise_op:wips_sw,
                                     agent.model.wnorm_ips_score_stepwise_mean_op:wips_sw_mean})
            print("[IPS Policy Evaluation @learn_count={}] ips={}, ips_stepwise={}, wnorm_ips={}, wnorm_ips_stepwise={}, wnorm_ips_stepwise_mean={}".format(
                learn_count, ips, ips_sw, wips, wips_sw, wips_sw_mean))

        if learn_count % 2000 == 0:
            # export saved model at any time
            # AssertionError will occur if the export_dir already exists.
            agent.export_saved_model(FLAGS.oss_path_save+"export_dir{}".format(learn_count))

    print("Done.", "num_epoch:", env.num_epoch, "learn_count:", learn_count,
          "total_cost:", time.time() - total_cost)


if __name__ == "__main__":
    main()
