import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dd_bandits import constants


def uncertainty_plots(
    n_arms: int,
    n_episodes: int,
    change_freq: int,
    df: pd.DataFrame,
    save_folder: str,
) -> None:

    fig = plt.figure(figsize=(12, 4 * n_arms))

    for arm in range(n_arms):
        plt.subplot(n_arms, 2, 2 * arm + 1)
        for ep in range(1, n_episodes + 1):
            plt.axvline(change_freq * ep, linestyle="--", color="k", lw=2)

        std_pattern = re.compile(
            f"{constants.ENSEMBLE_STD}_{constants.ARM}_{arm}_{constants.HEAD}_[0-9]"
        )
        mean_pattern = re.compile(
            f"{constants.ENSEMBLE_MEAN}_{constants.ARM}_{arm}_{constants.HEAD}_[0-9]"
        )

        stds_df = df.filter(regex=(std_pattern))
        means_df = df.filter(regex=(mean_pattern))
        dist_df = df[f"{constants.DISTRIBUTION_STD}_{arm}"]
        kl_average_df = df[f"{constants.AVERAGE_KL_DIV}_{arm}"]
        kl_max_df = df[f"{constants.MAX_KL_DIV}_{arm}"]
        ir_df = df[f"{constants.INFORMATION_RADIUS}_{arm}"]

        plt.plot(np.array(stds_df).mean(-1), zorder=5, lw=3)
        plt.plot(np.array(dist_df), lw=3, color="r")

        plt.tick_params(labelsize=14)
        _ = plt.title(f"Expected Uncertainty Arm {arm + 1}", fontsize=16)

        plt.subplot(n_arms, 2, 2 * arm + 2)
        for ep in range(1, n_episodes + 1):
            plt.axvline(change_freq * ep, linestyle="--", color="k", lw=2)

        plt.plot(
            np.array(means_df).std(-1),
            zorder=5,
            lw=3,
            label="mean std",
        )
        plt.plot(np.array(kl_average_df), zorder=5, lw=3, label="average KL")
        # plt.plot(np.array(kl_max_df), zorder=5, lw=3, label="max KL")
        plt.plot(
            np.array(ir_df),
            zorder=5,
            lw=3,
            label="inf radius",
        )
        plt.legend()
        plt.tick_params(labelsize=14)
        _ = plt.title(f"Unexpected Uncertainty {arm + 1}", fontsize=16)

    fig.savefig(os.path.join(save_folder, constants.UNCERTAINTY_PLOTS_PDF))
