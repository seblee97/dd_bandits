from typing import Dict, List, Union

import numpy as np
from dd_bandits import constants


class SelectAction:
    def __init__(self, config):
        if config.action_selection == constants.EPSILON_GREEDY:
            self._epsilon_computer = self._setup_epsilon_computer(config)
        self._select_action = self._setup_action_selection(config)

        self._latest_metrics: Dict[str, np.ndarray]
        self._estimator_group: List
        self._ucb_constant: Union[float, int]
        self._timestep: int = 0

        self._n_arms = config.n_arms

    def __call__(self, latest_metrics, estimator_group):
        self._timestep += 1
        self._estimator_group = estimator_group
        self._latest_metrics = latest_metrics
        return self._select_action()

    def _setup_action_selection(self, config):
        if config.action_selection == constants.EPSILON_GREEDY:
            return self._epsilon_greedy_action
        elif config.action_selection == constants.UCB:
            self._ucb_constant = config.ucb_constant
            return self._ucb_action
        elif config.action_selection == constants.THOMPSON:
            return self._thompson_action

    def _epsilon_greedy_action(self):
        epsilon = self._epsilon_computer()

        # epsilon-greedy action selection
        if np.random.random() < epsilon:
            action = np.random.randint(self._n_arms)
        else:
            action = np.argmax(
                [np.mean(eg.group_means) for eg in self._estimator_group]
            )

        return action, {constants.EPSILON: epsilon}

    def _ucb_action(self):
        action_counts = self._latest_metrics[constants.ACTION_COUNTS]
        ucb_values = [
            np.mean(eg.group_means)
            + self._ucb_constant * np.sqrt(np.log(self._timestep) / action_counts[a])
            for a, eg in enumerate(self._estimator_group)
        ]
        action = np.argmax(ucb_values)
        return action, {}

    def _thompson_action(self):
        pass

    def _setup_epsilon_computer(self, config):
        if config.eps_type == constants.CONSTANT:

            def eps_fn():
                return config.eps_value

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
