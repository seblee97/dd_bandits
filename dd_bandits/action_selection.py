import abc
from typing import Dict, List, Union

import numpy as np
from dd_bandits import constants


class SelectAction(abc.ABC):
    def __init__(self, config):
        self._latest_metrics: Dict[str, np.ndarray]
        self._estimator_group: List
        self._timestep: int = 0

        self._n_arms = config.n_arms

    def __call__(self, latest_metrics, estimator_group):
        self._timestep += 1
        self._estimator_group = estimator_group
        self._latest_metrics = latest_metrics
        return self._select_action()

    @abc.abstractmethod
    def _select_action(self):
        pass

    @abc.abstractmethod
    def update(self, action, reward):
        pass


class ThompsonSampling(SelectAction):
    def __init__(self, config):
        super().__init__(config=config)

        self._mus = np.zeros(self._n_arms)
        self._sigmas = np.zeros(self._n_arms)

    def _select_action(self):
        group_means = [eg.group_means for eg in self._estimator_group]
        group_stds = [eg.group_stds for eg in self._estimator_group]
        samples = [
            np.mean(
                [
                    np.random.normal(loc=mean, scale=std)
                    for mean, std in zip(means, stds)
                ]
            )
            for means, stds in zip(group_means, group_stds)
        ]
        # samples = [
        #     np.random.normal(loc=mean, scale=std)
        #     for mean, std in zip(self._mus, self._sigmas)
        # ]
        action = np.argmax(samples)
        return action, {}

    def update(self, action, reward):
        pass
        # self._mus[action]
        # self._sigmas[action]


class UCB(SelectAction):
    def __init__(self, config):
        super().__init__(config=config)
        self._ucb_constant: Union[float, int] = config.ucb_constant

    def _select_action(self):
        total_arm_rewards = self._latest_metrics[constants.TOTAL_ARM_REWARDS]
        action_counts = self._latest_metrics[constants.ACTION_COUNTS]
        empirical_reward_estimates = total_arm_rewards / action_counts

        ucb_values = [
            reward_estimate
            + self._ucb_constant
            * np.sqrt(2 * np.log(self._timestep) / action_counts[a])
            for a, reward_estimate in enumerate(empirical_reward_estimates)
        ]
        action = np.argmax(ucb_values)
        return action, {}

    def update(self, action, reward):
        pass


class DiscountedUCB(SelectAction):
    """Garivier & Moulines (2008)"""

    def __init__(self, config):
        super().__init__(config=config)
        self._ucb_gamma: Union[float, int] = config.ucb_gamma
        self._ucb_epsilon: Union[float, int] = config.ucb_epsilon
        self._ucb_constant: Union[float, int] = config.ucb_constant

    def _select_action(self):
        action_history = self._latest_metrics[constants.ACTION_HISTORY]
        reward_history = self._latest_metrics[constants.REWARD_HISTORY]

        discounted_counts = np.zeros(self._n_arms)
        discounted_values = np.zeros(self._n_arms)
        for i, history_positions in enumerate(action_history):
            for pos in history_positions:
                discounted_counts[i] += self._ucb_gamma ** (self._timestep - (pos + 1))
                discounted_values[i] += (
                    self._ucb_gamma ** (self._timestep - (pos + 1))
                    * reward_history[pos]
                )
        count_discounted_values = discounted_values / discounted_counts

        padding_discounting = (
            2
            * self._ucb_constant
            * np.sqrt(
                self._ucb_epsilon * np.log(sum(discounted_counts)) / discounted_counts
            )
        )
        ucb_values = count_discounted_values + padding_discounting
        action = np.argmax(ucb_values)

        return action, {}

    def update(self, action, reward):
        pass


class Softmax(SelectAction):
    def __init__(self, config):
        super().__init__(config=config)
        self._beta_computer = self._setup_beta_computer(config)

    def _setup_beta_computer(self, config):
        if config.beta_type == constants.CONSTANT:

            def beta_fn():
                return config.beta_value

        elif config.beta_type == constants.LINEAR_DECAY:

            initial_beta = config.initial_beta
            final_beta = config.final_beta
            beta_decay = config.beta_decay

            def beta_fn():
                return max([initial_beta - self._timestep * beta_decay, final_beta])

        elif config.beta_type == constants.MAX_STD_OF_MEAN:

            def beta_fn():
                computed_beta = np.min(
                    [1, np.max(self._latest_metrics[constants.STD_OF_MEAN])]
                )
                return 1 / np.max([config.minimum_beta, computed_beta])

        elif config.beta_type == constants.MEAN_STD_OF_MEAN:

            def beta_fn():
                computed_beta = np.min(
                    [1, np.mean(self._latest_metrics[constants.STD_OF_MEAN])]
                )
                return 1 / np.max([config.minimum_beta, computed_beta])

        elif config.beta_type == constants.MEAN_AVERAGE_KL:

            def beta_fn():
                computed_beta = np.min(
                    [1, np.mean(self._latest_metrics[constants.AVERAGE_KL])]
                )
                return 1 / np.max([config.minimum_beta, computed_beta])

        elif config.beta_type == constants.MEAN_MAX_KL:

            def beta_fn():
                computed_beta = np.min(
                    [1, np.mean(self._latest_metrics[constants.MAX_KL])]
                )
                return 1 / np.max([config.minimum_beta, computed_beta])

        elif config.beta_type == constants.MEAN_INFORMATION_RADIUS:

            def beta_fn():
                computed_beta = np.min(
                    [1, np.mean(self._latest_metrics[constants.INF_RADIUS])]
                )
                return 1 / np.max([config.minimum_beta, computed_beta])

        return beta_fn

    def _select_action(self):
        beta = self._beta_computer()

        max_value_estimates = np.array(
            [np.mean(eg.group_means) for eg in self._estimator_group]
        )
        exp_max_value_estimates = np.exp(max_value_estimates / beta)
        sotfmax_distribution = exp_max_value_estimates / sum(exp_max_value_estimates)

        action = np.random.choice(range(self._n_arms), p=sotfmax_distribution)

        return action, {constants.BETA: beta}

    def update(self, action, reward):
        pass


class EpsilonGreedy(SelectAction):
    def __init__(self, config):
        super().__init__(config=config)
        self._epsilon_computer = self._setup_epsilon_computer(config)

    def _setup_epsilon_computer(self, config):
        if config.eps_type == constants.CONSTANT:

            def eps_fn():
                return config.eps_value

        elif config.eps_type == constants.LINEAR_DECAY:

            initial_eps = config.initial_eps
            final_eps = config.final_eps
            eps_decay = config.eps_decay

            def eps_fn():
                return max([initial_eps - self._timestep * eps_decay, final_eps])

        elif config.eps_type == constants.MAX_STD_OF_MEAN:

            def eps_fn():
                computed_eps = np.min(
                    [1, np.max(self._latest_metrics[constants.STD_OF_MEAN])]
                )
                return np.max([config.minimum_eps, computed_eps])

        elif config.eps_type == constants.MEAN_STD_OF_MEAN:

            def eps_fn():
                computed_eps = np.min(
                    [1, np.mean(self._latest_metrics[constants.STD_OF_MEAN])]
                )
                return np.max([config.minimum_eps, computed_eps])

        elif config.eps_type == constants.MEAN_AVERAGE_KL:

            def eps_fn():
                computed_eps = np.min(
                    [1, np.mean(self._latest_metrics[constants.AVERAGE_KL])]
                )
                return np.max([config.minimum_eps, computed_eps])

        elif config.eps_type == constants.MEAN_MAX_KL:

            def eps_fn():
                computed_eps = np.min(
                    [1, np.mean(self._latest_metrics[constants.MAX_KL])]
                )
                return np.max([config.minimum_eps, computed_eps])

        elif config.eps_type == constants.MEAN_INFORMATION_RADIUS:

            def eps_fn():
                computed_eps = np.min(
                    [1, np.mean(self._latest_metrics[constants.INF_RADIUS])]
                )
                return np.max([config.minimum_eps, computed_eps])

        return eps_fn

    def _select_action(self):
        epsilon = self._epsilon_computer()

        # epsilon-greedy action selection
        if np.random.random() < epsilon:
            action = np.random.randint(self._n_arms)
        else:
            action = np.argmax(
                [np.mean(eg.group_means) for eg in self._estimator_group]
            )

        return action, {constants.EPSILON: epsilon}

    def update(self, action, reward):
        pass
