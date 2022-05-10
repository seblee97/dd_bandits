import itertools
from typing import Dict, List

import numpy as np
import pandas as pd
import utils
from dd_bandits import constants
from run_modes import base_runner


class Runner(base_runner.BaseRunner):
    def __init__(self, config, unique_id: str):
        self._estimator_group = [
            utils.EstimatorGroup(config.n_ensemble) for _ in range(config.n_arms)
        ]

        self._n_ensemble = config.n_ensemble
        self._n_episodes = config.n_episodes
        self._change_freq = config.change_freq
        self._n_arms = config.n_arms
        self._p_bootstrap = config.p_bootstrap
        self._batch_size = config.batch_size

        self._step_count: int
        self._data_index: int

        rng = np.random.RandomState(config.seed)
        self._means = [
            [utils.sample_mean(rng) for _ in range(self._n_ensemble)]
            for _ in range(self._n_episodes)
        ]
        self._scales = [
            [utils.sample_scale(rng) for _ in range(self._n_ensemble)]
            for _ in range(self._n_episodes)
        ]

        self._epsilon_computer = self._setup_epsilon_computer(config=config)
        self._lr_computer = self._setup_lr_computer(config=config)

        self._data_columns = self._setup_data_columns()

        super().__init__(config=config)

    def _get_data_columns(self):
        return list(self._data_columns.keys())

    def _setup_data_columns(self):
        data_column_names = [
            constants.ACTION_SELECTED,
            constants.EPSILON,
            constants.LEARNING_RATE,
            constants.REWARD,
            constants.MEAN_OPTIMAL_REWARD,
        ]

        for n_arm in range(self._n_arms):
            data_column_names.append(f"{constants.DISTRIBUTION_MEAN}_{n_arm}")
            data_column_names.append(f"{constants.DISTRIBUTION_STD}_{n_arm}")
            data_column_names.append(f"{constants.AVERAGE_KL_DIV}_{n_arm}")
            data_column_names.append(f"{constants.MAX_KL_DIV}_{n_arm}")
            data_column_names.append(f"{constants.INFORMATION_RADIUS}_{n_arm}")

        for (n_arm, e) in itertools.product(
            range(self._n_arms), range(self._n_ensemble)
        ):
            data_column_names.append(
                f"{constants.ENSEMBLE_MEAN}_{constants.ARM}_{n_arm}_{constants.HEAD}_{e}"
            )
            data_column_names.append(
                f"{constants.ENSEMBLE_STD}_{constants.ARM}_{n_arm}_{constants.HEAD}_{e}"
            )

        data_columns = {}
        for key in data_column_names:
            arr = np.empty(self._change_freq)
            arr[:] = np.NaN
            data_columns[key] = arr

        return data_columns

    def _setup_epsilon_computer(self, config):
        if config.eps_type == constants.CONSTANT:

            def eps_fn():
                return config.eps_value

        return eps_fn

    def _setup_lr_computer(self, config):
        if config.lr_type == constants.CONSTANT:

            def lr_fn():
                return config.lr_value

        return lr_fn

    def _log_trial(self, trial_index: int, logging_dict: Dict[str, float]) -> None:
        """Write scalars for all quantities collected in logging dictionary.

        Args:
            trial_index: current trial.
            logging_dict: dictionary of items to be logged collected during training.
        """
        for tag, scalar in logging_dict.items():
            self._data_logger.write_scalar(tag=tag, step=trial_index, scalar=scalar)

    def _checkpoint_data(self):
        self._data_logger.logger_data = pd.DataFrame.from_dict(self._data_columns)
        self._data_logger.checkpoint()
        self._data_columns = self._setup_data_columns()
        self._data_index = 0

    def train(self):

        self._step_count = 0
        self._data_index = 0

        for episode in range(self._n_episodes):

            self._logger.info(f"Beginning episode {episode}")

            means = self._means[episode]
            scales = self._scales[episode]

            for trial_index in range(self._change_freq):
                for n_arm in range(self._n_arms):
                    self._data_columns[f"{constants.DISTRIBUTION_MEAN}_{n_arm}"][
                        self._data_index
                    ] = means[n_arm]
                    self._data_columns[f"{constants.DISTRIBUTION_STD}_{n_arm}"][
                        self._data_index
                    ] = np.sqrt(scales[n_arm])
                self._trial(trial_index=trial_index, means=means, scales=scales)
                self._step_count += 1
                self._data_index += 1

            self._checkpoint_data()

    def _trial(self, trial_index: int, means: List[float], scales: List[float]):

        epsilon = self._epsilon_computer()

        # epsilon-greedy action selection
        if np.random.random() < epsilon:
            action = np.random.randint(self._n_arms)
        else:
            action = np.argmax(
                [np.mean(eg.group_means) for eg in self._estimator_group]
            )

        sampled_ind = np.where(np.random.random(self._n_ensemble) < self._p_bootstrap)[
            0
        ]

        if len(sampled_ind) == 0:
            sampled_ind = [np.random.randint(self._n_ensemble)]

        learning_rate = self._lr_computer()

        samples = self._estimator_group[action].update_subset(
            indices=sampled_ind,
            num_samples=self._batch_size,
            distribution_mean=means[action],
            distribution_std=np.sqrt(scales[action]),
            learning_rate=learning_rate,
        )

        reward = np.sum(samples)

        self._data_columns[constants.ACTION_SELECTED][self._data_index] = action
        self._data_columns[constants.EPSILON][self._data_index] = epsilon
        self._data_columns[constants.LEARNING_RATE][self._data_index] = learning_rate
        self._data_columns[constants.REWARD][self._data_index] = reward
        self._data_columns[constants.MEAN_OPTIMAL_REWARD][self._data_index] = (
            self._batch_size * len(sampled_ind) * max(means)
        )

        for n_arm in range(self._n_arms):

            kls = []

            dists = list(
                zip(
                    self._estimator_group[n_arm].group_means,
                    self._estimator_group[n_arm].group_stds,
                )
            )

            for ((mu_1, sigma_1), (mu_2, sigma_2)) in itertools.product(dists, dists):
                kl_12 = utils.kl_div(mu_1, mu_2, sigma_1, sigma_2)
                kls.append(kl_12)

            self._data_columns[f"{constants.AVERAGE_KL_DIV}_{n_arm}"][
                self._data_index
            ] = np.mean(kls)
            self._data_columns[f"{constants.MAX_KL_DIV}_{n_arm}"][
                self._data_index
            ] = np.max(kls)
            self._data_columns[f"{constants.INFORMATION_RADIUS}_{n_arm}"][
                self._data_index
            ] = utils.compute_information_radius(
                self._estimator_group[n_arm].group_means,
                self._estimator_group[n_arm].group_stds,
            )

            for e, (e_mean, e_std) in enumerate(
                zip(
                    self._estimator_group[n_arm].group_means,
                    self._estimator_group[n_arm].group_stds,
                )
            ):
                self._data_columns[
                    f"{constants.ENSEMBLE_MEAN}_{constants.ARM}_{n_arm}_{constants.HEAD}_{e}"
                ][self._data_index] = e_mean
                self._data_columns[
                    f"{constants.ENSEMBLE_STD}_{constants.ARM}_{n_arm}_{constants.HEAD}_{e}"
                ][self._data_index] = e_mean
