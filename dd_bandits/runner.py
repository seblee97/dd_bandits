import itertools
from typing import Dict, List

import numpy as np
import pandas as pd
from dd_bandits import action_selection, constants, plot_functions, utils
from run_modes import base_runner


class Runner(base_runner.BaseRunner):
    def __init__(self, config, unique_id: str):

        self._n_ensemble = config.n_ensemble
        self._n_episodes = config.n_episodes
        self._change_freq = config.change_freq
        self._n_arms = config.n_arms
        self._p_bootstrap = config.p_bootstrap
        self._batch_size = config.batch_size
        self._default_lr = config.default_lr
        self._default_epsilon = config.default_eps

        self._latest_metrics = {
            constants.STD_OF_MEAN: np.zeros(self._n_arms),
            constants.MEAN_OF_STD: np.zeros(self._n_arms),
            constants.AVERAGE_KL: np.zeros(self._n_arms),
            constants.MAX_KL: np.zeros(self._n_arms),
            constants.INF_RADIUS: np.zeros(self._n_arms),
            constants.ACTION_COUNTS: np.zeros(self._n_arms),
        }

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

        self._action_selector = self._setup_action_selection(config=config)
        self._optimiser = self._setup_optimiser(config=config)
        self._lr_computer = self._setup_lr_computer(config=config)

        self._data_columns = self._setup_data_columns()

        self._estimator_group = [
            utils.EstimatorGroup(
                optimiser=self._optimiser, total_num_estimators=config.n_ensemble
            )
            for _ in range(config.n_arms)
        ]

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

    def _setup_action_selection(self, config):
        if config.action_selection == constants.EPSILON_GREEDY:
            return action_selection.EpsilonGreedy(config=config)
        elif config.action_selection == constants.UCB:
            return action_selection.UCB(config=config)
        elif config.action_selection == constants.THOMPSON:
            return action_selection.ThompsonSampling(config=config)

    def _setup_optimiser(self, config):
        if config.optimiser == constants.SGD:
            return utils.SGD()
        elif config.optimiser == constants.ADAM:
            return utils.Adam(
                alpha=config.alpha,
                beta_1=config.beta_1,
                beta_2=config.beta_2,
                epsilon=config.epsilon,
            )
        elif config.optimiser == constants.RMS_PROP:
            return utils.RMSProp(
                eta=config.eta, gamma=config.gamma, epsilon=config.epsilon
            )

    def _setup_lr_computer(self, config):
        if config.lr_type == constants.CONSTANT:

            def lr_fn(action: int):
                return config.lr_value

        elif config.lr_type == constants.LINEAR_DECAY:

            def lr_fn(action: int):
                return max(
                    [
                        config.initial_lr - self._step_count * config.lr_decay,
                        config.final_lr,
                    ]
                )

        elif config.lr_type == constants.ACTION_MEAN_OF_STD:

            def lr_fn(action: int):
                lr = config.factor * self._latest_mean_of_std[action]
                if lr == 0:
                    lr = self._default_lr
                return lr

        elif config.lr_type == constants.MEAN_MEAN_OF_STD:

            def lr_fn(action: int):
                lr = config.factor * np.mean(
                    self._latest_metrics[constants.MEAN_OF_STD]
                )
                if lr == 0:
                    lr = self._default_lr
                return lr

        elif config.lr_type == constants.MEAN_STD_OF_MEAN:

            def lr_fn(action: int):
                lr = config.factor * np.mean(
                    self._latest_metrics[constants.STD_OF_MEAN]
                )
                if lr == 0:
                    lr = self._default_lr
                return lr

        elif config.lr_type == constants.UNCERTAINTY_FRACTION:

            def lr_fn(action: int):
                epistemic_uncertainty = np.mean(
                    self._latest_metrics[constants.MEAN_OF_STD]
                )
                aleatoric_uncertainty = np.mean(
                    self._latest_metrics[constants.AVERAGE_KL]
                )
                lr = (
                    config.factor
                    * epistemic_uncertainty
                    / (epistemic_uncertainty + aleatoric_uncertainty + 0.0001)
                )
                if lr == 0:
                    lr = self._default_lr
                return lr

        return lr_fn

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

        action, metrics = self._action_selector(
            latest_metrics=self._latest_metrics, estimator_group=self._estimator_group
        )
        for key, metric in metrics.items():
            self._data_columns[key][self._data_index] = metric

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
        self._action_selector.update(action, reward)
        optimal_reward = self._batch_size * len(sampled_ind) * max(means)

        self._data_columns[constants.ACTION_SELECTED][self._data_index] = action
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

        self._latest_metrics[constants.ACTION_COUNTS][action] += 1

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

            inf_radius = utils.compute_information_radius(
                self._estimator_group[n_arm].group_means,
                self._estimator_group[n_arm].group_stds,
            )
            self._data_columns[f"{constants.INFORMATION_RADIUS}_{n_arm}"][
                self._data_index
            ] = inf_radius
            self._latest_metrics[constants.INF_RADIUS][n_arm] = inf_radius

            self._latest_metrics[constants.AVERAGE_KL][n_arm] = np.mean(kls)
            self._latest_metrics[constants.MAX_KL][n_arm] = np.max(kls)

            mean_of_std = np.mean(self._estimator_group[n_arm].group_stds)
            mean_of_mean = np.mean(self._estimator_group[n_arm].group_means)
            std_of_std = np.std(self._estimator_group[n_arm].group_stds)
            std_of_mean = np.std(self._estimator_group[n_arm].group_means)

            self._data_columns[f"{constants.MEAN_OF_STD}_{n_arm}"][
                self._data_index
            ] = mean_of_std
            self._latest_metrics[constants.MEAN_OF_STD][n_arm] = mean_of_std

            self._data_columns[f"{constants.MEAN_OF_MEAN}_{n_arm}"][
                self._data_index
            ] = mean_of_mean
            self._data_columns[f"{constants.STD_OF_STD}_{n_arm}"][
                self._data_index
            ] = std_of_std

            self._data_columns[f"{constants.STD_OF_MEAN}_{n_arm}"][
                self._data_index
            ] = std_of_mean
            self._latest_metrics[constants.STD_OF_MEAN][n_arm] = std_of_mean

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
                ][self._data_index] = e_std

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
