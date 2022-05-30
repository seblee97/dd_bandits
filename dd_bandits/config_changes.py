import itertools

eps_constants = [0.1, 0.5, 0.9]
lr_constants = [0.01, 0.1, 1]
lr_modulations = [0.1, 0.05, 0.2]
action_selections = ["ucb", "epsilon_greedy"]
lr_types = ["mean_mean_of_std", "uncertainty_fraction", "linear_decay"]

eps_types = [
    "mean_std_of_mean",
    "mean_max_kl",
    "mean_average_kl",
    "mean_information_radius",
]

CONSTANT_CONFIG_CHANGES = {
    f"lr_constant_{value_lr}_eps_constant_{value_eps}_{action_selection}": [
        {
            "optimiser": "sgd",
            "action_selection": action_selection,
            "learning_rate": {"type": "constant", "constant": {"value": value_lr}},
            "epsilon": {"type": "constant", "constant": {"value": value_eps}},
        }
    ]
    for value_lr, value_eps, action_selection in itertools.product(
        lr_constants, eps_constants, action_selections
    )
}

ADAPTIVE_SGD_CONFIG_CHANGES = {
    f"lr_{lr_type}_{modulation}_eps_{eps_type}_{action_selection}": [
        {
            "optimiser": "sgd",
            "action_selection": action_selection,
            "learning_rate": {"type": lr_type, "modulate": {"factor": modulation}},
            "epsilon": {"type": eps_type},
        }
    ]
    for lr_type, modulation, eps_type, action_selection in itertools.product(
        lr_types, lr_modulations, eps_types, action_selections
    )
}

ADAPTIVE_RMS_CONFIG_CHANGES = {
    f"rms_eps_{eps_type}_{action_selection}": [
        {
            "optimiser": "rms_prop",
            "action_selection": action_selection,
            "epsilon": {"type": eps_type},
        }
    ]
    for eps_type, action_selection in itertools.product(eps_types, action_selections)
}

CONFIG_CHANGES = {
    **CONSTANT_CONFIG_CHANGES,
    **ADAPTIVE_SGD_CONFIG_CHANGES,
    **ADAPTIVE_RMS_CONFIG_CHANGES,
}
