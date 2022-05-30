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
        # group_means = [eg.group_means for eg in self._estimator_group]
        # group_stds = [eg.group_stds for eg in self._estimator_group]
        # samples = [
        #     [np.random.normal(loc=mean, scale=std) for mean, std in zip(means, stds)]
        #     for means, stds in zip(group_means, group_stds)
        # ]
        samples = [
            np.random.normal(loc=mean, scale=std)
            for mean, std in zip(self._mus, self._sigmas)
        ]
        action = np.argmax(np.mean(samples, axis=1))
        return action, {}

    def update(self, action, reward):
        self._mus[action]
        self._sigmas[action]


class UCB(SelectAction):
    def __init__(self, config):
        super().__init__(config=config)
        self._ucb_constant: Union[float, int] = config.ucb_constant

    def _select_action(self):
        action_counts = self._latest_metrics[constants.ACTION_COUNTS]
        ucb_values = [
            np.mean(eg.group_means)
            + self._ucb_constant * np.sqrt(np.log(self._timestep) / action_counts[a])
            for a, eg in enumerate(self._estimator_group)
        ]
        action = np.argmax(ucb_values)
        return action, {}

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
