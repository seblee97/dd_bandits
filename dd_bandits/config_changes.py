import itertools

eps_constants = [0.1, 0.5, 0.9]
beta_constants = [1, 10, 100, 1000]
lr_constants = [0.01, 0.1, 1]
ucb_constants = [1, 10, 20, 50, 100, 1000]
ducb_gammas = [0.8, 0.9, 0.95, 0.99]
ducb_eps = [0.25, 0.5, 0.75, 0.9]
lr_modulations = [0.1, 0.05, 0.2]
beta_modulations = [0.01, 0.1, 1, 10, 100]
action_selections = ["ucb", "epsilon_greedy"]
lr_types = ["mean_mean_of_std", "uncertainty_fraction", "linear_decay"]

eps_types = [
    "mean_std_of_mean",
    "mean_max_kl",
    "mean_average_kl",
    "mean_information_radius",
]

beta_types = [
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

UCB_CONFIG_CHANGES = {
    f"ucb_{ucb}_lr_{lr}": [
        {
            "optimiser": "sgd",
            "action_selection": "ucb",
            "ucb": {"ucb_constant": ucb},
            "learning_rate": {"type": "constant", "constant": {"value": lr}},
        }
    ]
    for ucb, lr in itertools.product(ucb_constants, lr_constants)
}

DISCOUNTED_UCB_CONFIG_CHANGES = {
    f"discounted_ucb_{const}_{gamma}_{eps}_lr_{lr}": [
        {
            "optimiser": "sgd",
            "action_selection": "discounted_ucb",
            "discounted_ucb": {
                "ucb_constant": const,
                "ucb_gamma": gamma,
                "ucb_epsilon": eps,
            },
            "learning_rate": {"type": "constant", "constant": {"value": lr}},
        }
    ]
    for const, gamma, eps, lr in itertools.product(
        ucb_constants, ducb_gammas, ducb_eps, lr_constants
    )
}

SOFTMAX_CONFIG_CHANGES = {
    f"softmax_{beta}_lr_{lr}": [
        {
            "optimiser": "sgd",
            "action_selection": "softmax",
            "beta": {"type": "constant", "constant": {"value": beta}},
            "learning_rate": {"type": "constant", "constant": {"value": lr}},
        }
    ]
    for beta, lr in itertools.product(beta_constants, lr_constants)
}

DD_EXPLORATION_CONFIG_CHANGES = {
    f"doya_dayu_{beta_type}_{beta_mod}_lr_{lr}": [
        {
            "optimiser": "sgd",
            "action_selection": "softmax",
            "beta": {"type": beta_type, "modulate": {"beta_factor": beta_mod}},
            "learning_rate": {"type": "constant", "constant": {"value": lr}},
        }
    ]
    for beta_type, beta_mod, lr in itertools.product(
        beta_types, beta_modulations, lr_constants
    )
}

CONFIG_CHANGES = {
    **CONSTANT_CONFIG_CHANGES,
    **ADAPTIVE_SGD_CONFIG_CHANGES,
    **ADAPTIVE_RMS_CONFIG_CHANGES,
}
