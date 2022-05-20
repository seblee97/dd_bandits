import itertools

eps_constants = [0.1, 0.5, 0.9]
lr_constants = [0.01, 0.1, 1]
lr_modulations = [0.1, 0.05, 0.2, 0.5]
lr_types = ["action_mean_of_std", "mean_mean_of_std"]

eps_types = ["max_std_of_mean", "mean_std_of_mean"]

ADAPTIVE_CONFIG_CHANGES = {
    f"lr_{lr_type}_{modulation}_eps_{eps_type}": [
        {
            "learning_rate": {"type": lr_type, "modulate": {"factor": modulation}},
            "epsilon": {"type": eps_type},
        }
    ]
    for lr_type, modulation, eps_type in itertools.product(
        lr_types, lr_modulations, eps_types
    )
}

CONSTANT_LR_CONFIG_CHANGES = {
    f"lr_constant_{value}_eps_{eps_type}": [
        {
            "learning_rate": {"type": "constant", "constant": {"value": value}},
            "epsilon": {"type": eps_type},
        }
    ]
    for value, eps_type in itertools.product(lr_constants, eps_types)
}

CONSTANT_EPS_CONFIG_CHANGES = {
    f"lr_{lr_type}_{modulation}_eps_constant_{value}": [
        {
            "learning_rate": {"type": lr_type, "modulate": {"factor": modulation}},
            "epsilon": {"type": "constant", "constant": {"value": value}},
        }
    ]
    for lr_type, modulation, value in itertools.product(
        lr_types, lr_modulations, eps_constants
    )
}

CONSTANT_CONFIG_CHANGES = {
    f"lr_constant_{value_lr}_eps_constant_{value_eps}": [
        {
            "learning_rate": {"type": "constant", "constant": {"value": value_lr}},
            "epsilon": {"type": "constant", "constant": {"value": value_eps}},
        }
    ]
    for value_lr, value_eps in itertools.product(lr_constants, eps_constants)
}

CONFIG_CHANGES = {
    **ADAPTIVE_CONFIG_CHANGES,
    **CONSTANT_LR_CONFIG_CHANGES,
    **CONSTANT_EPS_CONFIG_CHANGES,
    **CONSTANT_CONFIG_CHANGES,
}
