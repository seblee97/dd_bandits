import itertools
from typing import Dict, List

import numpy as np
import pandas as pd
from dd_bandits import constants, plot_functions, utils
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
        self._default_lr = config.default_lr
        self._default_epsilon = config.default_eps

        self._latest_std_of_mean = np.zeros(self._n_arms)
        self._latest_mean_of_std = np.zeros(self._n_arms)

        self._step_count: int
        self._data_index: int

        rng = np.random.RandomState(config.seed)
        self._means = [
            [
                utils.sample_mean(rng, config.distribution_mean_range)
                for _ in range(self._n_arms)
            ]
            for _ in range(self._n_episodes)
        ]
        self._stds = [
            [
                utils.sample_scale(rng, config.distribution_std_range)
                for _ in range(self._n_arms)
            ]
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
            # constants.ACTION_SELECTED,
            constants.EPSILON,
            constants.LEARNING_RATE,
            constants.REWARD,
            constants.MEAN_OPTIMAL_REWARD,
            constants.REGRET,
        ]

        for n_arm in range(self._n_arms):
            data_column_names.append(f"{constants.DISTRIBUTION_MEAN}_{n_arm}")
            data_column_names.append(f"{constants.DISTRIBUTION_STD}_{n_arm}")
            data_column_names.append(f"{constants.AVERAGE_KL_DIV}_{n_arm}")
            data_column_names.append(f"{constants.MAX_KL_DIV}_{n_arm}")
            data_column_names.append(f"{constants.INFORMATION_RADIUS}_{n_arm}")
            data_column_names.append(f"{constants.MEAN_OF_STD}_{n_arm}")
            data_column_names.append(f"{constants.MEAN_OF_MEAN}_{n_arm}")
            data_column_names.append(f"{constants.STD_OF_STD}_{n_arm}")
            data_column_names.append(f"{constants.STD_OF_MEAN}_{n_arm}")

        # for (n_arm, e) in itertools.product(
        #     range(self._n_arms), range(self._n_ensemble)
        # ):
        #     data_column_names.append(
        #         f"{constants.ENSEMBLE_MEAN}_{constants.ARM}_{n_arm}_{constants.HEAD}_{e}"
        #     )
        #     data_column_names.append(
        #         f"{constants.ENSEMBLE_STD}_{constants.ARM}_{n_arm}_{constants.HEAD}_{e}"
        #     )

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

        elif config.eps_type == constants.MAX_STD_OF_MEAN:

            def eps_fn():
                return np.min([1, np.max(self._latest_std_of_mean)])

        elif config.eps_type == constants.MEAN_STD_OF_MEAN:

            def eps_fn():
                return np.min([1, np.mean(self._latest_std_of_mean)])

        return eps_fn

    def _setup_lr_computer(self, config):
        if config.lr_type == constants.CONSTANT:

            def lr_fn(action: int):
                return config.lr_value

        elif config.lr_type == constants.ACTION_MEAN_OF_STD:

            def lr_fn(action: int):
                lr = config.factor * self._latest_mean_of_std[action]
                if lr == 0:
                    lr = self._default_lr
                return lr

        elif config.lr_type == constants.MEAN_MEAN_OF_STD:

            def lr_fn(action: int):
                lr = config.factor * np.mean(self._latest_mean_of_std)
                if lr == 0:
                    lr = self._default_lr
                return lr

        elif config.lr_type == constants.MEAN_STD_OF_MEAN:

            def lr_fn(action: int):
                lr = config.factor * np.mean(self._latest_std_of_mean)
                if lr == 0:
                    lr = self._default_lr
                return lr

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

        self._reward = 0
        self._mean_optimal_reward = 0

        for episode in range(self._n_episodes):

            self._logger.info(f"Beginning episode {episode}")

            means = self._means[episode]
            stds = self._stds[episode]

            for trial_index in range(self._change_freq):
                for n_arm in range(self._n_arms):
                    self._data_columns[f"{constants.DISTRIBUTION_MEAN}_{n_arm}"][
                        self._data_index
                    ] = means[n_arm]
                    self._data_columns[f"{constants.DISTRIBUTION_STD}_{n_arm}"][
                        self._data_index
                    ] = stds[n_arm]
                self._trial(trial_index=trial_index, means=means, stds=stds)
                self._step_count += 1
                self._data_index += 1

            self._checkpoint_data()

    def _trial(self, trial_index: int, means: List[float], stds: List[float]):

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

        learning_rate = self._lr_computer(action=action)

        samples = self._estimator_group[action].update_subset(
            indices=sampled_ind,
            num_samples=self._batch_size,
            distribution_mean=means[action],
            distribution_std=stds[action],
            learning_rate=learning_rate,
        )

        reward = np.sum(samples)
        optimal_reward = self._batch_size * len(sampled_ind) * max(means)

        # self._data_columns[constants.ACTION_SELECTED][self._data_index] = action
        self._data_columns[constants.EPSILON][self._data_index] = epsilon
        self._data_columns[constants.LEARNING_RATE][self._data_index] = learning_rate

        self._data_columns[constants.REWARD][self._data_index] = reward
        self._data_columns[constants.MEAN_OPTIMAL_REWARD][
            self._data_index
        ] = optimal_reward

        self._reward += reward
        self._mean_optimal_reward += optimal_reward

        self._data_columns[constants.REGRET][self._data_index] = (
            self._mean_optimal_reward - self._reward
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

            mean_of_std = np.mean(self._estimator_group[n_arm].group_stds)
            mean_of_mean = np.mean(self._estimator_group[n_arm].group_means)
            std_of_std = np.std(self._estimator_group[n_arm].group_stds)
            std_of_mean = np.std(self._estimator_group[n_arm].group_means)

            self._data_columns[f"{constants.MEAN_OF_STD}_{n_arm}"][
                self._data_index
            ] = mean_of_std
            self._latest_mean_of_std[n_arm] = mean_of_std

            self._data_columns[f"{constants.MEAN_OF_MEAN}_{n_arm}"][
                self._data_index
            ] = mean_of_mean
            self._data_columns[f"{constants.STD_OF_STD}_{n_arm}"][
                self._data_index
            ] = std_of_std

            self._data_columns[f"{constants.STD_OF_MEAN}_{n_arm}"][
                self._data_index
            ] = std_of_mean
            self._latest_std_of_mean[n_arm] = std_of_mean

            # for e, (e_mean, e_std) in enumerate(
            #     zip(
            #         self._estimator_group[n_arm].group_means,
            #         self._estimator_group[n_arm].group_stds,
            #     )
            # ):
            #     self._data_columns[
            #         f"{constants.ENSEMBLE_MEAN}_{constants.ARM}_{n_arm}_{constants.HEAD}_{e}"
            #     ][self._data_index] = e_mean
            #     self._data_columns[
            #         f"{constants.ENSEMBLE_STD}_{constants.ARM}_{n_arm}_{constants.HEAD}_{e}"
            #     ][self._data_index] = e_std

    def post_process(self):
        df = self._data_logger.load_data()
        plot_functions.uncertainty_plots(
            n_arms=self._n_arms,
            n_episodes=self._n_episodes,
            change_freq=self._change_freq,
            df=df,
            save_folder=self._checkpoint_path,
        )
        plot_functions.epsilon_plot(df=df, save_folder=self._checkpoint_path)
        plot_functions.lr_plot(df=df, save_folder=self._checkpoint_path)
        plot_functions.regret_plot(df=df, save_folder=self._checkpoint_path)
