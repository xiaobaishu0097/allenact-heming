from typing import Sequence, Union

import torch.nn as nn

from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.utils.experiment_utils import Builder, TrainingPipeline
from allenact.base_abstractions.task import TaskSampler
from allenact_plugins.ithor_plugin.ithor_sensors import (
    GoalObjectTypeThorSensor,
    RGBSensorThor,
)
from allenact_plugins.robothor_plugin.robothor_task_samplers import ObjectNavTaskSampler
from projects.objectnav_baselines.experiments.robothor.objectnav_robothor_base import (
    ObjectNavRoboThorBaseConfig,
)
from projects.objectnav_baselines.detector_mixins import ResNetDETRPreprocessGRUActorCriticMixin
from projects.objectnav_baselines.mixins import ObjectNavPPOMixin


class ObjectNavRoboThorRGBPPOExperimentConfig(ObjectNavRoboThorBaseConfig):
    """An Object Navigation experiment configuration in RoboThor with RGB
    input."""

    SENSORS = [
        RGBSensorThor(
            height=ObjectNavRoboThorBaseConfig.SCREEN_SIZE,
            width=ObjectNavRoboThorBaseConfig.SCREEN_SIZE,
            use_resnet_normalization=True,
            uuid="rgb_lowres",
        ),
        GoalObjectTypeThorSensor(
            object_types=ObjectNavRoboThorBaseConfig.TARGET_TYPES,
        ),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.preprocessing_and_model = ResNetDETRPreprocessGRUActorCriticMixin(
            sensors=self.SENSORS,
            resnet_type="RN18",
            screen_size=self.SCREEN_SIZE,
            goal_sensor_type=GoalObjectTypeThorSensor,
        )

    def training_pipeline(self, **kwargs) -> TrainingPipeline:
        return ObjectNavPPOMixin.training_pipeline(
            auxiliary_uuids=[],
            multiple_beliefs=False,
            advance_scene_rollout_period=self.ADVANCE_SCENE_ROLLOUT_PERIOD,
        )

    def preprocessors(self) -> Sequence[Union[Preprocessor, Builder[Preprocessor]]]:
        return self.preprocessing_and_model.preprocessors()

    def create_model(self, **kwargs) -> nn.Module:
        return self.preprocessing_and_model.create_model(
            num_actions=self.ACTION_SPACE.n, **kwargs
        )

    @classmethod
    def make_sampler_fn(cls, **kwargs) -> TaskSampler:
        return ObjectNavTaskSampler(**kwargs)

    @classmethod
    def tag(cls):
        return "ObjectNav-RoboTHOR-RGB-ResNet18DETRGRU-TASKSAMPLER-DDPPO"
