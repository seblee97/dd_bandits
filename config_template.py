import constants
from config_manager import config_field, config_template


def get_template():
    template_class = ConfigTemplate()
    return template_class.base_template


class ConfigTemplate:
    def __init__(self):
        pass

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
                config_field.Field(name=constants.LEARNING_RATE, types=[int, float]),
                config_field.Field(
                    name=constants.EPSILON,
                    types=[int, float],
                    requirements=[lambda x: x >= 0 and x <= 1],
                ),
                config_field.Field(name=constants.MODULATE_LR, types=[type(None), str]),
                config_field.Field(
                    name=constants.MODULATE_EPS, types=[type(None), str]
                ),
                config_field.Field(
                    name=constants.DISTRIBUTION_MEAN_RANGE, types=[list]
                ),
                config_field.Field(
                    name=constants.DISTRIBUTION_SCALE_RANGE, types=[list]
                ),
            ],
            nested_templates=[],
        )
