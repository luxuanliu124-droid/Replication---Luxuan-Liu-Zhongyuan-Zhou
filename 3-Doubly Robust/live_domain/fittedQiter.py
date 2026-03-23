import pickle
import numpy as np
from live_domain.live_simulator.live import LiveTreatment
from sklearn.externals import joblib
from live_domain.batch_policy_learning import BatchPolicyLearning
from live_domain.live_simulator.batch_data_processing import BatchDataProcessing #changed on Oct 2, 2019


class FittedQIteration():
    """FittedQIteration is an implementation of the Fitted Q-Iteration algorithm of Ernst, Geurts, Wehenkel (2005).
    This class allows the use of a variety of regression algorithms, provided by scikits-learn, to be used for
    representing the Q-value function. Additionally, different basis functions can be applied to the features before
    being passed to the regressors, including trivial, fourier, tile coding, and radial basis functions."""
    def __init__(self, gamma = 0.98, iterations = 400, K = 10, num_consumers = 30000, preset_params = None, ins = None,\
        episode_length = 200, cache=False):
        """Inits the Fitted Q-Iteration planner with discount factor, instantiated model learner, and additional parameters.

        Args:
        model: The model learner object
        gamma=0.98: The discount factor for the domain
        **kwargs: Additional parameters for use in the class.
        """
        self.gamma = gamma
        self.iterations = iterations
        self.K = K
        # Set up regressor
        self.scenario = 'SIMULATE'  # SIMULATOR  USEBATCH #changed on Oct 2, 2019
        self.task = LiveTreatment(self.scenario)
        self.action_dim = 50
        self.state_dim = 100
        self.num_consumers = num_consumers
        self.eps = 0.0
        self.samples = None
        self.preset_params = preset_params
        self.ins = ins
        self.episode_length = episode_length
        self.cache = cache
        self.bpl = BatchPolicyLearning(None)
        self.bdp = BatchDataProcessing(self.scenario, None)  # need to initialize the class before calling the functions below


    def run_episode(self, i_episode,eps = 0.0, track = False):
        """Run an episode on the environment (and train Q function if modelfree)."""
        if self.scenario == 'USEBATCH': #changed on Oct 2, 2019
            ep_list=bdp.get_episode(i_episode) #changed on Oct 2, 2019
            return ep_list
        else: #changed on Oct 2, 2019
            self.task.reset(**self.preset_params)
            state = self.task.observe()
            # task is done after max_task_examples timesteps or when the agent enters a terminal state
            ep_list = []
            action_list = []
            ep_reward = 0
            while not self.task.is_done(episode_length=self.episode_length):
                action_prob = self.policy(state, eps)
                action = np.random.multinomial(1,action_prob).argmax() # changed on Oct 1, 2019
                action_list.append(action)
                reward, next_state = self.task.perform_action(state, action, **self.preset_params)
                if track:
                    ep_list.append(np.array([state, action, reward, next_state]))
                state = next_state
                ep_reward += (reward*self.gamma**self.task.t)
            return ep_list

    def policy(self, state, eps = 0.0):
        """Get the action under the current plan policy for the given state.

        Args:
        state: The array of state features

        Returns:
        The current greedy action under the planned policy for the given state. If no plan has been formed,
        return a random action.
        """

        if np.random.rand(1) < eps:
            return np.random.dirichlet(np.ones(self.action_dim)) # changed on Oct 1, 2019
        else:
            return self.bpl.batch_policy_learning(state)

    def bpolicy(self, state): #changed on Oct 2, 2019
        """Get the action under the behavioral policy for the given state.

        Args:
        state: The array of state features

        Returns:
        Behavioral Policy
        """

        return self.bdp.policy_prob(state)