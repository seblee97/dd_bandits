from typing import Dict, List, Union

from config_manager import base_configuration
from dd_bandits import config_template


class Config(base_configuration.BaseConfiguration):
    """Wrapper for base configuration

    Implements a specific validate configuration method for
    non-trivial associations that need checking in config.
    """

    def __init__(self, config: Union[str, Dict], changes: List[Dict] = []) -> None:
        base_template = config_template.get_template()
        super().__init__(
            configuration=config,
            template=base_template,
            changes=changes,
        )

        self._validate_config()

    def _validate_config(self) -> None:
        """Check for non-trivial associations in config.

        Raises:
            AssertionError: if any rules are broken by config.
        """
        pass

    def _maybe_reconfigure(self, property_name: str) -> None:
        pass
