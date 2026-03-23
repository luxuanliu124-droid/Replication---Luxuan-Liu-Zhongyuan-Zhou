#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import enum
import logging
from functools import reduce
from typing import Dict, Optional, Tuple, Union

import gym
import ml.rl.test.gym.pomdp  # noqa
import numpy as np
import torch
from gym import Env
from ml.rl.test.base.utils import only_continuous_normalizer
from ml.rl.test.environment.environment import Environment
from ml.rl.training.on_policy_predictor import OnPolicyPredictor
from live_env import Live_Env


logger = logging.getLogger(__name__)


class ModelType(enum.Enum):
    CONTINUOUS_ACTION = "continuous"
    SOFT_ACTOR_CRITIC = "soft_actor_critic"
    TD3 = "td3"
    PYTORCH_DISCRETE_DQN = "pytorch_discrete_dqn"
    PYTORCH_PARAMETRIC_DQN = "pytorch_parametric_dqn"
    CEM = "cross_entropy_method"


class EnvType(enum.Enum):
    DISCRETE_ACTION = "discrete"
    CONTINUOUS_ACTION = "continuous"
    UNKNOWN = "unknown"


class LiveEnvironment():
    def __init__(
        self,
        epsilon=0,
        softmax_policy=False,
        gamma=0.99,
        epsilon_decay=1,
        minimum_epsilon=None,
        random_seed: Optional[int] = None,
    ):
        """
        Creates an OpenAIGymEnvironment object.

        :param gymenv: String identifier for desired environment or environment
            object itself.
        :param epsilon: Fraction of the time the agent should select a random
            action during training.
        :param softmax_policy: 1 to use softmax selection policy or 0 to use
            max q selection.
        :param gamma: Discount rate
        :param epsilon_decay: How much to decay epsilon over each iteration in training.
        :param minimum_epsilon: Lower bound of epsilon.
        :param random_seed: The random seed for the environment
        """
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.minimum_epsilon = minimum_epsilon
        self.softmax_policy = softmax_policy
        self.gamma = gamma
        self.action_type = "Discrete"
        self.state_dim = 24
        self.action_dim = 50
        self.img = False
        self._create_env()

        if not self.img:
            assert self.state_dim > 0
            self.state_features = [str(sf) for sf in range(self.state_dim)]
        if self.action_type == "Discrete":
            assert self.action_dim > 0
            self.actions = [str(a + self.state_dim) for a in range(self.action_dim)]

    @property
    def normalization(self):
        if self.img:
            return None
        else:
            return only_continuous_normalizer(
                list(range(self.state_dim)),
                0,
                self.state_dim,
            )

    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay
        if self.minimum_epsilon is not None:
            self.epsilon = max(self.epsilon, self.minimum_epsilon)

    def _create_env(self):
        """
        Creates a gym environment object and checks if it is supported. We
        support environments that supply Box(x, ) state representations and
        require Discrete(y) or Box(y,) action inputs.

        :param gymenv: String identifier for desired environment or environment
            object itself.
        """
        self.env = Live_Env()
        self.EnvType = "LIVE"
        
    def reset(self):
        init_state = self.env.reset()
        assert len(init_state) == self.state_dim
        return init_state

    def step(self, action):
        res = self.env.step(action)
        next_state = res[0]
        assert len(next_state) == self.state_dim
        return res

    def policy(self, state):
        """
        Selects the next action.
        :param state: State to evaluate predictor's policy on.
        """
        '''state: torch.Tensor'''
        assert len(state.size()) == 1
        # Convert state to batch of size 1
        state = state.unsqueeze(0)
        action, action_probability = self.env.get_action_from_state(state)
        return action, action_probability

    def run_ep_n_times(
        self,
        n,
        predictor: Optional[OnPolicyPredictor],
        max_steps=None,
        test=False,
        render=False,
    ):
        """
        Runs an episode of the environment n times and returns the average
        sum of rewards.

        :param n: Number of episodes to average over.
        :param predictor: OnPolicyPredictor object whose policy to
            follow. If set to None, use a random policy
        :param max_steps: Max number of timesteps before ending episode.
        :param test: Whether o6r not to bypass an epsilon-greedy selection policy.
        :param render: Whether or not to render the episode.
        """
        reward_sum = 0.0
        discounted_reward_sum = 0.0
        for _ in range(n):
            ep_rew_sum, ep_raw_discounted_sum = self.run_episode(
                predictor, max_steps, test, render
            )
            reward_sum += ep_rew_sum
            discounted_reward_sum += ep_raw_discounted_sum
        avg_rewards = round(reward_sum / n, 2)
        avg_discounted_rewards = round(discounted_reward_sum / n, 2)
        return avg_rewards, avg_discounted_rewards

    def transform_state(self, state: np.ndarray) -> torch.Tensor:
        torch_state = torch.from_numpy(state).float()
        return torch_state

    def run_episode(
        self,
        predictor: Optional[OnPolicyPredictor],
        max_steps=None,
        test=False,
        render=False,
    ):
        """
        Runs an episode of the environment and returns the sum of rewards
        experienced in the episode. For evaluation purposes.

        :param predictor: OnPolicyPredictor object whose policy to
            follow. If set to None, use a random policy.
        :param max_steps: Max number of timesteps before ending episode.
        :param test: Whether or not to bypass an epsilon-greedy selection policy.
        :param render: Whether or not to render the episode.
        """
        terminal = False
        next_state_numpy = self.reset()
        next_action, _ = self.policy(torch.from_numpy(next_state_numpy))
        reward_sum = 0.0
        discounted_reward_sum = 0
        num_steps_taken = 0

        while not terminal:
            logger.debug(
                f"OpenAIGym: {num_steps_taken}-th step, state: {next_state_numpy}"
            )
            action = next_action
            action_index = action
            next_state_numpy, reward, terminal, _ = self.step(action_index)
            logger.debug(
                f"OpenAIGym: take action {action_index}, reward: {reward}, terminal: {terminal}"
            )
            num_steps_taken += 1

            next_action, _ = self.policy(torch.tensor(next_state_numpy))
            reward_sum += float(reward)
            discounted_reward_sum += reward * self.gamma ** (num_steps_taken - 1)

            if max_steps and num_steps_taken >= max_steps:
                break

        self.reset()
        return reward_sum, discounted_reward_sum

    def _process_state(self, raw_state: np.ndarray) -> Dict:
        processed_state = {}
        for i in range(self.state_dim):
            processed_state[i] = raw_state[i]
        return processed_state

    def sample_policy(self, state, use_continuous_action: bool, epsilon: float = 0.0):
        """
        Sample a random action
        Return the raw action which can be fed into env.step(), the processed
            action which can be uploaded to Hive, and action probability
        """
        raw_action = self.env.action_space.sample()  # type: ignore

        if self.action_type == EnvType.DISCRETE_ACTION:
            action_probability = 1.0 / self.action_dim
            if not use_continuous_action:
                return raw_action, str(self.state_dim + raw_action), action_probability
            action_vec = {self.state_dim + raw_action: 1}
            return raw_action, action_vec, action_probability

        if self.action_type == EnvType.CONTINUOUS_ACTION:
            # action_probability is the probability density of multi-variate
            # uniform distribution
            range_each_dim = (
                self.env.observation_space.high  # type: ignore
                - self.env.observation_space.low  # type: ignore
            )
            action_probability = 1.0 / reduce((lambda x, y: x * y), range_each_dim)
            action_vec = {}
            for i in range(self.action_dim):
                action_vec[self.state_dim + i] = raw_action[i]
            return raw_action, action_vec, action_probability


    def possible_actions(
        self,
        state,
        terminal: bool = False,
        ignore_terminal=False,
        use_continuous_action: bool = False,
        **kwargs,
    ):
        # possible actions will not be used in algorithms dealing with
        # continuous actions, so just return an empty list
        if terminal or self.action_type == EnvType.CONTINUOUS_ACTION:
            return []
        if not use_continuous_action:
            return self.actions
        possible_actions = []
        for i in range(self.action_dim):
            action_vec = {self.state_dim + i: 1}
            possible_actions.append(action_vec)
        return possible_actions
