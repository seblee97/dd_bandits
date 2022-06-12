import copy
from typing import List, Union

import numpy as np


class Estimator:
    def __init__(
        self,
        optimiser,
        mean_init: Union[float, int] = 0,
        std_init: Union[float, int] = 1,
    ):
        self._mean_estimate = np.random.normal(mean_init, std_init / 10)
        self._std_estimate = std_init

        self._optimiser = optimiser

    @property
    def mean_estimate(self):
        return self._mean_estimate

    @property
    def std_estimate(self):
        return self._std_estimate

    def update(self, new_sample, learning_rate):
        new_sample_mean = np.mean(new_sample)
        new_sample_logvar = np.log(np.var(new_sample))
        mean_gradient = 2 * (self._mean_estimate - new_sample_mean)
        logvar_gradient = 2 * (np.log(self._std_estimate**2) - new_sample_logvar)
        # delta = self._mean_estimate - new_sample
        # delta_bar = self._std_estimate**2 - delta**2

        update_step = self._optimiser.update(
            # gradient=np.array([delta.mean(), delta_bar.mean()]),
            gradient=np.array([mean_gradient, logvar_gradient]),
            learning_rate=learning_rate,
        )

        self._mean_estimate += update_step[0]
        logvar = np.log(self._std_estimate**2) + update_step[1]
        # if var <= 0:
        #     import pdb

        #     pdb.set_trace()
        self._std_estimate = np.sqrt(np.exp(logvar))


class EstimatorGroup:
    def __init__(self, total_num_estimators: int, optimiser):
        self._total_num_estimators = total_num_estimators
        self._estimators = [
            Estimator(copy.deepcopy(optimiser)) for _ in range(total_num_estimators)
        ]

        self._mean_diffs = []

    def update_subset(
        self,
        indices,
        num_samples,
        distribution_mean,
        distribution_std,
        learning_rate,
    ):

        samples = np.random.normal(
            distribution_mean, distribution_std, size=(len(indices), num_samples)
        )

        for sample, estimator_index in zip(samples, indices):
            self._estimators[estimator_index].update(
                sample, learning_rate=learning_rate
            )

        subset_means = [self._estimators[m].mean_estimate for m in indices]
        non_subset_means = [
            self._estimators[m].mean_estimate
            for m in range(self._total_num_estimators)
            if m in indices
        ]

        mean_diff = np.mean(subset_means) - np.mean(non_subset_means)
        self._mean_diffs.append(mean_diff)

        return samples

    @property
    def group_means(self):
        return [estimator.mean_estimate for estimator in self._estimators]

    @property
    def group_stds(self):
        return [estimator.std_estimate for estimator in self._estimators]

    @property
    def mean_diffs(self):
        return self._mean_diffs


class SGD:
    def __init__(self):
        pass

    def update(self, gradient: np.ndarray, learning_rate):
        return -learning_rate * gradient


class RMSProp:
    def __init__(
        self,
        eta: Union[float, int],
        gamma: Union[float, int],
        epsilon: Union[float, int],
        dimension: int = 2,
    ):
        self._eta = eta
        self._gamma = gamma
        self._epsilon = epsilon

        self._eg2 = np.zeros(dimension)

    def update(self, gradient, learning_rate):
        self._eg2 = self._gamma * self._eg2 + (1 - self._gamma) * gradient**2
        return -self._eta * gradient / np.sqrt(self._eg2 + self._epsilon)


class Adam:
    def __init__(
        self,
        alpha: Union[float, int],
        beta_1: Union[float, int],
        beta_2: Union[float, int],
        epsilon: Union[float, int],
        dimension: int = 2,
    ):
        self._alpha = alpha
        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._epsilon = epsilon

        self._step_count = 0

        self._m = np.zeros(dimension)
        self._v = np.zeros(dimension)

    def update(self, gradient, learning_rate):
        self._step_count += 1
        self._m = self._beta_1 * self._m + (1 - self._beta_1) * gradient
        self._v = self._beta_2 * self._v + (1 - self._beta_2) * gradient**2
        m_hat = self._m / (1 - self._beta_1**self._step_count)
        v_hat = self._v / (1 - self._beta_2**self._step_count)
        return -self._alpha * m_hat / (np.sqrt(v_hat) + self._epsilon)


def sample_mean(rng, mean_range: List[Union[float, int]]):
    return rng.uniform(mean_range[0], mean_range[1])


def sample_scale(rng, std_range: List[Union[float, int]]):
    return rng.uniform(std_range[0], std_range[1])


def kl_div(mu_1, mu_2, sigma_1, sigma_2):
    return (
        np.log(sigma_2 / sigma_1)
        + (sigma_1**2 + (mu_1 - mu_2) ** 2) / (2 * sigma_2**2)
        - 0.5
    )


def compute_information_radius(means, stds):
    average_mean = np.mean(means)
    average_var = np.mean(np.array(stds) ** 2)

    kls = []

    for (mean, std) in zip(means, stds):
        kls.append(kl_div(mean, average_mean, std, np.sqrt(average_var)))

    return np.mean(kls)
