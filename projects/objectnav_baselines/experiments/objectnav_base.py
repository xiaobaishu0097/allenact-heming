from abc import ABC
from typing import Optional, Sequence, Union

import os
import yaml

from allenact.base_abstractions.experiment_config import ExperimentConfig
from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.base_abstractions.sensor import Sensor
from allenact.utils.experiment_utils import Builder


class ObjectNavBaseConfig(ExperimentConfig, ABC):
    """The base object navigation configuration file."""

    STEP_SIZE = 0.25
    ROTATION_DEGREES = 30.0
    VISIBILITY_DISTANCE = 1.0
    STOCHASTIC = True
    HORIZONTAL_FIELD_OF_VIEW = 79

    CAMERA_WIDTH = 400
    CAMERA_HEIGHT = 300
    SCREEN_SIZE = 224
    MAX_STEPS = 500

    ADVANCE_SCENE_ROLLOUT_PERIOD: Optional[int] = None
    SENSORS: Sequence[Sensor] = []

    def __init__(self, agent_config: Optional[str] = None):
        if agent_config is not None and os.path.exists(agent_config):
            with open(agent_config, "r") as file:
                loaded_settings = yaml.safe_load(file)

            for key, value in loaded_settings.items():
                if hasattr(self, key) and getattr(self, key) != value:
                    setattr(self, key, value)

        self.REWARD_CONFIG = {
            "step_penalty": -0.01,
            "goal_success_reward": 10.0,
            "failed_stop_reward": 0.0,
            "shaping_weight": 1.0,
        }

    def preprocessors(self) -> Sequence[Union[Preprocessor, Builder[Preprocessor]]]:
        return tuple()
