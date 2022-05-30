from config_manager import config_field, config_template
from dd_bandits import constants


def get_template():
    template_class = ConfigTemplate()
    return template_class.base_template


class ConfigTemplate:
    def __init__(self):
        pass

    _constant_lr_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.VALUE,
                key=constants.LR_VALUE,
                types=[int, float],
                requirements=[lambda x: x > 0],
            )
        ],
        level=[constants.LEARNING_RATE, constants.CONSTANT],
        dependent_variables=[constants.LR_TYPE],
        dependent_variables_required_values=[[constants.CONSTANT]],
    )

    _linear_decay_lr_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.INITIAL_LR,
                types=[int, float],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.FINAL_LR,
                types=[int, float],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.LR_DECAY,
                types=[int, float],
                requirements=[lambda x: x > 0],
            ),
        ],
        level=[constants.LEARNING_RATE, constants.LINEAR_DECAY],
        dependent_variables=[constants.LR_TYPE],
        dependent_variables_required_values=[[constants.LINEAR_DECAY]],
    )

    _modulate_mean_of_std_lr_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.FACTOR,
                types=[int, float],
            )
        ],
        level=[constants.LEARNING_RATE, constants.MODULATE],
        dependent_variables=[constants.LR_TYPE],
        dependent_variables_required_values=[
            [
                constants.ACTION_MEAN_OF_STD,
                constants.MEAN_MEAN_OF_STD,
                constants.MEAN_STD_OF_MEAN,
                constants.UNCERTAINTY_FRACTION,
            ]
        ],
    )

    _adam_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.ALPHA,
                types=[int, float],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.BETA_1,
                types=[int, float],
                requirements=[lambda x: x >= 0 and x <= 1],
            ),
            config_field.Field(
                name=constants.BETA_2,
                types=[int, float],
                requirements=[lambda x: x >= 0 and x <= 1],
            ),
            config_field.Field(
                name=constants.EPSILON,
                types=[int, float],
                requirements=[lambda x: x > 0],
            ),
        ],
        level=[constants.ADAM],
        dependent_variables=[constants.OPTIMISER],
        dependent_variables_required_values=[[constants.ADAM]],
    )

    _rms_prop_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.ETA,
                types=[int, float],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.GAMMA,
                types=[int, float],
                requirements=[lambda x: x >= 0 and x <= 1],
            ),
            config_field.Field(
                name=constants.EPSILON,
                types=[int, float],
                requirements=[lambda x: x > 0],
            ),
        ],
        level=[constants.RMS_PROP],
        dependent_variables=[constants.OPTIMISER],
        dependent_variables_required_values=[[constants.RMS_PROP]],
    )

    _learning_rate_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.TYPE,
                key=constants.LR_TYPE,
                types=[str],
                requirements=[
                    lambda x: x
                    in [
                        constants.CONSTANT,
                        constants.LINEAR_DECAY,
                        constants.ACTION_MEAN_OF_STD,
                        constants.MEAN_MEAN_OF_STD,
                        constants.MEAN_STD_OF_MEAN,
                        constants.UNCERTAINTY_FRACTION,
                    ]
                ],
            ),
            config_field.Field(
                name=constants.DEFAULT_LR,
                types=[int, float],
                requirements=[lambda x: x > 0],
            ),
        ],
        level=[constants.LEARNING_RATE],
        nested_templates=[
            _constant_lr_template,
            _linear_decay_lr_template,
            _modulate_mean_of_std_lr_template,
        ],
    )

    _constant_eps_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.VALUE,
                key=constants.EPS_VALUE,
                types=[int, float],
                requirements=[lambda x: x > 0],
            )
        ],
        level=[constants.EPSILON, constants.CONSTANT],
        dependent_variables=[constants.EPS_TYPE],
        dependent_variables_required_values=[[constants.CONSTANT]],
    )

    _linear_decay_eps_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.INITIAL_EPS,
                types=[int, float],
                requirements=[lambda x: x >= 0 and x <= 1],
            ),
            config_field.Field(
                name=constants.FINAL_EPS,
                types=[int, float],
                requirements=[lambda x: x >= 0 and x <= 1],
            ),
            config_field.Field(
                name=constants.EPS_DECAY,
                types=[int, float],
                requirements=[lambda x: x > 0 and x <= 1],
            ),
        ],
        level=[constants.EPSILON, constants.LINEAR_DECAY],
        dependent_variables=[constants.EPS_TYPE],
        dependent_variables_required_values=[[constants.LINEAR_DECAY]],
    )

    _epsilon_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.TYPE,
                key=constants.EPS_TYPE,
                types=[str],
                requirements=[
                    lambda x: x
                    in [
                        constants.CONSTANT,
                        constants.LINEAR_DECAY,
                        constants.MAX_STD_OF_MEAN,
                        constants.MEAN_STD_OF_MEAN,
                        constants.MEAN_AVERAGE_KL,
                        constants.MEAN_INFORMATION_RADIUS,
                        constants.MEAN_MAX_KL,
                    ]
                ],
            ),
            config_field.Field(
                name=constants.DEFAULT_EPS,
                types=[int, float],
                requirements=[lambda x: x >= 0 and x <= 1],
            ),
            config_field.Field(
                name=constants.MINIMUM_EPS,
                types=[float, int],
                requirements=[lambda x: x >= 0 and x <= 1],
            ),
        ],
        level=[constants.EPSILON],
        nested_templates=[_constant_eps_template, _linear_decay_eps_template],
    )

    _ucb_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.UCB_CONSTANT,
                types=[float, int],
                requirements=[lambda x: x > 0],
            )
        ],
        level=[constants.UCB],
        dependent_variables=[constants.ACTION_SELECTION],
        dependent_variables_required_values=[[constants.UCB]],
    )

    @property
    def base_template(self):
        return config_template.Template(
            fields=[
                config_field.Field(name=constants.SEED, types=[int]),
                config_field.Field(name=constants.N_ARMS, types=[int]),
                config_field.Field(name=constants.N_ENSEMBLE, types=[int]),
                config_field.Field(
                    name=constants.P_BOOTSTRAP,
                    types=[float, int],
                    requirements=[lambda x: x >= 0 or x <= 1],
                ),
                config_field.Field(name=constants.CHANGE_FREQ, types=[int]),
                config_field.Field(name=constants.BATCH_SIZE, types=[int]),
                config_field.Field(name=constants.N_EPISODES, types=[int]),
                config_field.Field(
                    name=constants.DISTRIBUTION_MEAN_RANGE, types=[list]
                ),
                config_field.Field(name=constants.DISTRIBUTION_STD_RANGE, types=[list]),
                config_field.Field(
                    name=constants.ACTION_SELECTION,
                    types=[str],
                    requirements=[
                        lambda x: x
                        in [constants.EPSILON_GREEDY, constants.UCB, constants.THOMPSON]
                    ],
                ),
                config_field.Field(
                    name=constants.OPTIMISER,
                    types=[str],
                    requirements=[
                        lambda x: x
                        in [constants.SGD, constants.ADAM, constants.RMS_PROP]
                    ],
                ),
            ],
            nested_templates=[
                self._adam_template,
                self._rms_prop_template,
                self._learning_rate_template,
                self._epsilon_template,
                self._ucb_template,
            ],
        )
