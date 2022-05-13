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

    _learning_rate_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.TYPE,
                key=constants.LR_TYPE,
                types=[str],
                requirements=[lambda x: x in [constants.CONSTANT]],
            )
        ],
        level=[constants.LEARNING_RATE],
        nested_templates=[_constant_lr_template],
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

    _epsilon_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.TYPE,
                key=constants.EPS_TYPE,
                types=[str],
                requirements=[lambda x: x in [constants.CONSTANT]],
            )
        ],
        level=[constants.EPSILON],
        nested_templates=[_constant_eps_template],
    )

    @property
    def base_template(self):
        return config_template.Template(
            fields=[
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
            ],
            nested_templates=[self._learning_rate_template, self._epsilon_template],
        )
