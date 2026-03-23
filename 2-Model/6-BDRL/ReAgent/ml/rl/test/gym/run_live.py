#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import argparse
import json
import logging
import pickle
import random
import sys
from typing import Any, Dict, Optional

import numpy as np
import torch
from ml.rl.json_serialize import json_to_object
from ml.rl.parameters import (
    CEMParameters,
    CNNParameters,
    ContinuousActionModelParameters,
    DiscreteActionModelParameters,
    FeedForwardParameters,
    MDNRNNParameters,
    LiveParameters,
    LiveRunDetails,
    OptimizerParameters,
    RainbowDQNParameters,
    RLParameters,
    SACModelParameters,
    SACTrainingParameters,
    TD3ModelParameters,
    TD3TrainingParameters,
    TrainingParameters,
)
from ml.rl.test.base.utils import write_lists_to_csv
from ml.rl.test.gym.live_environment import (
    EnvType,
    ModelType,
    LiveEnvironment,
)
from ml.rl.test.gym.live_memory_pool import LiveMemoryPool
from ml.rl.training.on_policy_predictor import (
    CEMPlanningPredictor,
    ContinuousActionOnPolicyPredictor,
    DiscreteDQNOnPolicyPredictor,
    OnPolicyPredictor,
    ParametricDQNOnPolicyPredictor,
)
from ml.rl.training.rl_dataset import RLDataset
from ml.rl.training.rl_trainer_pytorch import RLTrainer
from ml.rl.workflow.transitional import (
    create_dqn_trainer_from_params,
)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split


logger = logging.getLogger(__name__)


def dict_to_np(d, np_size, key_offset):
    x = np.zeros(np_size, dtype=np.float32)
    for key in d:
        x[key - key_offset] = d[key]
    return x


def dict_to_torch(d, np_size, key_offset):
    x = torch.zeros(np_size)
    for key in d:
        x[key - key_offset] = d[key]
    return x


def get_possible_actions(live_env, model_type, terminal):
    if model_type == ModelType.PYTORCH_DISCRETE_DQN.value: 
        possible_next_actions = None
        possible_next_actions_mask = torch.tensor(
            [0 if terminal else 1 for __ in range(live_env.action_dim)]
        )
    elif model_type == ModelType.PYTORCH_PARAMETRIC_DQN.value:
        possible_next_actions = torch.eye(live_env.action_dim)
        possible_next_actions_mask = torch.tensor(
            [0 if terminal else 1 for __ in range(live_env.action_dim)]
        )
    elif model_type in (
        ModelType.CONTINUOUS_ACTION.value,
        ModelType.SOFT_ACTOR_CRITIC.value,
        ModelType.TD3.value,
        ModelType.CEM.value,
    ):
        possible_next_actions = None
        possible_next_actions_mask = None
    else:
        possible_next_actions = None
        possible_next_actions_mask = torch.tensor(
            [0 if terminal else 1 for __ in range(live_env.action_dim)]
        )
    return possible_next_actions, possible_next_actions_mask


def create_epsilon(offline_train, rl_parameters, params):
    if offline_train:
        # take random actions during data collection
        epsilon = 1.0
    else:
        epsilon = rl_parameters.epsilon
    epsilon_decay, minimum_epsilon = 1.0, None
    if params.run_details.epsilon_decay is not None:
        epsilon_decay = params.run_details.epsilon_decay
    minimum_epsilon = params.run_details.minimum_epsilon
    return epsilon, epsilon_decay, minimum_epsilon


def create_replay_buffer(
    env, params, model_type, offline_train, path_to_pickled_transitions
):
    """
    Train on transitions generated from a random policy live or
    read transitions from a pickle file and load into replay buffer.
    """
    replay_buffer = LiveMemoryPool(params.max_replay_memory_size)
    return replay_buffer


def train(
    live_env,
    offline_train,
    replay_buffer,
    model_type,
    trainer,
    predictor,
    test_run_name,
    score_bar,
    run_details: LiveRunDetails,
    save_timesteps_to_dataset=None,
    start_saving_from_score=None,
    bcq_imitator_hyperparams=None,
    reward_shape_func=None,
):
    return train_live_online_rl(
        live_env,
        replay_buffer,
        model_type,
        trainer,
        predictor,
        test_run_name,
        score_bar,
        run_details.num_episodes,
        run_details.max_steps,
        run_details.train_every_ts,
        run_details.train_after_ts,
        run_details.test_every_ts,
        run_details.test_after_ts,
        run_details.num_train_batches,
        run_details.avg_over_num_episodes,
        run_details.render,
        save_timesteps_to_dataset,
        start_saving_from_score,
        run_details.solved_reward_threshold,
        run_details.max_episodes_to_run_after_solved,
        run_details.stop_training_after_solved,
        reward_shape_func,
    )



def train_live_online_rl(
    live_env,
    replay_buffer,
    model_type,
    trainer,
    predictor,
    test_run_name,
    score_bar,
    num_episodes,
    max_steps,
    train_every_ts,
    train_after_ts,
    test_every_ts,
    test_after_ts,
    num_train_batches,
    avg_over_num_episodes,
    render,
    save_timesteps_to_dataset,
    start_saving_from_score,
    solved_reward_threshold,
    max_episodes_to_run_after_solved,
    stop_training_after_solved,
    reward_shape_func,
):
    """Train off of dynamic set of transitions generated on-policy."""
    total_timesteps = 0
    avg_reward_history, timestep_history = [], []
    best_episode_score_seen = -1e10
    episodes_since_solved = 0
    solved = False
    policy_id = 0

    for i in range(num_episodes):

        terminal = False
        next_state = live_env.transform_state(live_env.reset())
        next_action, next_action_probability = live_env.policy(next_state)

        reward_sum = 0
        ep_timesteps = 0

        while not terminal:
            state = next_state
            action = next_action
            action_probability = next_action_probability

            # Get possible actions
            possible_actions, _ = get_possible_actions(live_env, model_type, terminal)

            timeline_format_action, gym_action = _format_action_for_log_and_gym(
                action, live_env.action_type, model_type
            )

            next_state, reward, terminal, _ = live_env.step(gym_action)

            next_state = live_env.transform_state(next_state)

            if reward_shape_func:
                reward = reward_shape_func(next_state, ep_timesteps)

            ep_timesteps += 1
            total_timesteps += 1
            next_action, next_action_probability = live_env.policy(
                next_state
            )
            reward_sum += reward

            (possible_actions, possible_actions_mask) = get_possible_actions(
                live_env, model_type, False
            )

            # Get possible next actions
            (possible_next_actions, possible_next_actions_mask) = get_possible_actions(
                live_env, model_type, terminal
            )

            replay_buffer.insert_into_memory(
                state,
                action,
                reward,
                next_state,
                next_action,
                terminal,
                possible_next_actions,
                possible_next_actions_mask,
                1,
                possible_actions,
                possible_actions_mask,
                policy_id,
            )

            if save_timesteps_to_dataset and (
                start_saving_from_score is None
                or best_episode_score_seen >= start_saving_from_score
            ):
                save_timesteps_to_dataset.insert(
                    mdp_id=i,
                    sequence_number=ep_timesteps - 1,
                    state=state,
                    action=action,
                    timeline_format_action=timeline_format_action,
                    action_probability=action_probability,
                    reward=reward,
                    next_state=next_state,
                    next_action=next_action,
                    terminal=terminal,
                    possible_next_actions=possible_next_actions,
                    possible_next_actions_mask=possible_next_actions_mask,
                    time_diff=1,
                    possible_actions=possible_actions,
                    possible_actions_mask=possible_actions_mask,
                    policy_id=policy_id,
                )

            # Training loop
            if (
                total_timesteps % train_every_ts == 0
                and total_timesteps > train_after_ts
                and replay_buffer.size >= trainer.minibatch_size
                and not (stop_training_after_solved and solved)
            ):
                for _ in range(num_train_batches):
                    samples = replay_buffer.sample_memories(
                        trainer.minibatch_size, model_type
                    )
                    samples.set_device(trainer.device)
                    trainer.train(samples)
                    # Every time we train, the policy changes
                    policy_id += 1

            # Evaluation loop
            if total_timesteps % test_every_ts == 0 and total_timesteps > test_after_ts:
                avg_rewards, avg_discounted_rewards = live_env.run_ep_n_times(
                    avg_over_num_episodes, predictor, test=True, max_steps=max_steps
                )
                if avg_rewards > best_episode_score_seen:
                    best_episode_score_seen = avg_rewards

                if (
                    solved_reward_threshold is not None
                    and best_episode_score_seen >= solved_reward_threshold
                ):
                    solved = True

                avg_reward_history.append(avg_rewards)
                timestep_history.append(total_timesteps)
                logger.info(
                    "Achieved an average reward score of {}, discounted reward of {}, over {} evaluations."
                    " Total episodes: {}, total timesteps: {}.".format(
                        avg_rewards, avg_discounted_rewards, avg_over_num_episodes, i + 1, total_timesteps
                    )
                )
                if score_bar is not None and avg_rewards > score_bar:
                    logger.info(
                        "Avg. reward history for {}: {}".format(
                            test_run_name, avg_reward_history
                        )
                    )
                    return (
                        avg_reward_history,
                        timestep_history,
                        trainer,
                        predictor,
                        live_env,
                    )

            if max_steps and ep_timesteps >= max_steps:
                break

        # Always eval on last episode
        if i == num_episodes - 1:
            avg_rewards, avg_discounted_rewards = live_env.run_ep_n_times(
                avg_over_num_episodes, predictor, test=True, max_steps=max_steps
            )
            avg_reward_history.append(avg_rewards)
            timestep_history.append(total_timesteps)
            logger.info(
                "Achieved an average reward score of {}, discounted reward {}, over {} evaluations."
                " Total episodes: {}, total timesteps: {}.".format(
                    avg_rewards, avg_discounted_rewards, avg_over_num_episodes, i + 1, total_timesteps
                )
            )

        if solved:
            live_env.epsilon = live_env.minimum_epsilon
        else:
            live_env.decay_epsilon()

        if i % 50 == 0:
            logger.info(
                "Online RL episode {}, total_timesteps {}".format(i, total_timesteps)
            )

    logger.info(
        "Avg. reward history for {}: {}".format(test_run_name, avg_reward_history)
    )
    return avg_reward_history, timestep_history, trainer, predictor, live_env


def main(args):
    parser = argparse.ArgumentParser(
        description="Train a RL net to play in an OpenAI Gym environment."
    )
    parser.add_argument("-p", "--parameters", help="Path to JSON parameters file.")
    parser.add_argument(
        "-s",
        "--score-bar",
        help="Bar for averaged tests scores.",
        type=float,
        default=None,
    )
    parser.add_argument(
        "-l",
        "--log_level",
        help="If set, use logging level specified (debug, info, warning, error, "
        "critical). Else defaults to info.",
        default="info",
    )
    parser.add_argument(
        "-f",
        "--file_path",
        help="If set, save all collected samples as an RLDataset to this file.",
        default=None,
    )
    parser.add_argument(
        "-e",
        "--start_saving_from_score",
        type=int,
        help="If file_path is set, start saving episodes after this score is hit.",
        default=None,
    )
    parser.add_argument(
        "-r",
        "--results_file_path",
        help="If set, save evaluation results to file.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--offline_train",
        action="store_true",
        help="If set, collect data using a random policy then train RL offline.",
    )
    parser.add_argument(
        "--path_to_pickled_transitions",
        help="Path to saved transitions to load into replay buffer.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--seed",
        help="Seed for the test (numpy, torch, and gym).",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--use_gpu",
        help="Use GPU, if available; set the device with CUDA_VISIBLE_DEVICES",
        action="store_true",
    )

    args = parser.parse_args(args)

    if args.log_level not in ("debug", "info", "warning", "error", "critical"):
        raise Exception("Logging level {} not valid level.".format(args.log_level))
    else:
        logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    assert (
        not args.path_to_pickled_transitions or args.offline_train
    ), "path_to_pickled_transitions is provided so you must run offline training"

    with open(args.parameters, "r") as f:
        params = json_to_object(f.read(), LiveParameters)

    if args.use_gpu:
        assert torch.cuda.is_available(), "CUDA requested but not available"
        params = params._replace(use_gpu=True)

    dataset = RLDataset(args.file_path) if args.file_path else None

    reward_history, iteration_history, trainer, predictor, env = run_gym(
        params,
        args.offline_train,
        args.score_bar,
        args.seed,
        dataset,
        args.start_saving_from_score,
        args.path_to_pickled_transitions,
    )

    if dataset:
        dataset.save()
        logger.info("Saving dataset to {}".format(args.file_path))
        final_score_exploit, _ = env.run_ep_n_times(
            params.run_details.avg_over_num_episodes, predictor, test=True
        )
        final_score_explore, _ = env.run_ep_n_times(
            params.run_details.avg_over_num_episodes, predictor, test=False
        )
        logger.info(
            "Final policy scores {} with epsilon={} and {} with epsilon=0 over {} eps.".format(
                final_score_explore,
                env.epsilon,
                final_score_exploit,
                params.run_details.avg_over_num_episodes,
            )
        )

    if args.results_file_path:
        write_lists_to_csv(args.results_file_path, reward_history, iteration_history)
    return reward_history


def run_gym(
    params: LiveParameters,
    offline_train,
    score_bar,
    seed=None,
    save_timesteps_to_dataset=None,
    start_saving_from_score=None,
    path_to_pickled_transitions=None,
    warm_trainer=None,
    reward_shape_func=None,
):
    use_gpu = params.use_gpu
    logger.info("Running gym with params")
    logger.info(params)
    assert params.rl is not None
    rl_parameters = params.rl

    env_type = params.env
    model_type = params.model_type

    epsilon, epsilon_decay, minimum_epsilon = create_epsilon(
        offline_train, rl_parameters, params
    )
    '''
    env = OpenAIGymEnvironment(
        env_type,
        epsilon,
        rl_parameters.softmax_policy,
        rl_parameters.gamma,
        epsilon_decay,
        minimum_epsilon,
        seed,
    )
    '''
    env = LiveEnvironment()

    replay_buffer = create_replay_buffer(
        env, params, model_type, offline_train, path_to_pickled_transitions
    )

    trainer = warm_trainer if warm_trainer else create_trainer(params, env)
    predictor = create_predictor(trainer, model_type, use_gpu, env.action_dim)

    return train(
        env,
        offline_train,
        replay_buffer,
        model_type,
        trainer,
        predictor,
        "{} test run".format(env_type),
        score_bar,
        params.run_details,
        save_timesteps_to_dataset=save_timesteps_to_dataset,
        start_saving_from_score=start_saving_from_score,
        reward_shape_func=reward_shape_func,
    )


def create_trainer(params: LiveParameters, env: LiveEnvironment):
    use_gpu = params.use_gpu
    model_type = params.model_type
    print(model_type)
    assert params.rl is not None
    rl_parameters = params.rl
   
    assert params.training is not None
    training_parameters = params.training
    assert params.rainbow is not None
    
    discrete_trainer_params = DiscreteActionModelParameters(
        actions=env.actions,
        rl=rl_parameters,
        training=training_parameters,
        rainbow=params.rainbow,
        evaluation=params.evaluation,
    )
    trainer = create_dqn_trainer_from_params(
        discrete_trainer_params, env.normalization, use_gpu
    )

    
    return trainer


def _format_action_for_log_and_gym(action, env_type, model_type):
    if env_type == EnvType.DISCRETE_ACTION:
        action_index = torch.argmax(action).item()
        if model_type == ModelType.PYTORCH_DISCRETE_DQN.value:
            return str(action_index), int(action_index)
        else:
            return action.tolist(), int(action_index)
    return action.tolist(), action.tolist()


def create_predictor(trainer, model_type, use_gpu, action_dim=None):
    predictor = DiscreteDQNOnPolicyPredictor(trainer, action_dim, use_gpu)
    return predictor


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logging.getLogger().setLevel(logging.INFO)
    from ml.rl import debug_on_error

    debug_on_error.start()
    args = sys.argv
    main(args[1:])
