from typing import List, Union

import numpy as np


class Estimator:
    def __init__(
        self, mean_init: Union[float, int] = 0, std_init: Union[float, int] = 1
    ):
        self._mean_estimate = mean_init
        self._std_estimate = std_init

    @property
    def mean_estimate(self):
        return self._mean_estimate

    @property
    def std_estimate(self):
        return self._std_estimate

    def update(self, new_sample, learning_rate):
        # print("PRE", self._std_estimate)
        delta = new_sample - self._mean_estimate
        self._mean_estimate += learning_rate * delta.mean()

        delta_bar = delta**2 - self._std_estimate**2
        var = self._std_estimate**2 + learning_rate * delta_bar.mean()
        self._std_estimate = np.sqrt(var)


class EstimatorGroup:
    def __init__(self, total_num_estimators: int):
        self._total_num_estimators = total_num_estimators
        self._estimators = [Estimator() for _ in range(total_num_estimators)]

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
