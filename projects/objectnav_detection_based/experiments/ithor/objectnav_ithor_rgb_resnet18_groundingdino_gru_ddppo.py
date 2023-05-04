from typing import Sequence, Union

import torch.nn as nn

from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.utils.experiment_utils import Builder, TrainingPipeline
from allenact_plugins.ithor_plugin.ithor_sensors import (
    GoalObjectTypeThorSensor,
    RGBSensorThor,
)
from projects.objectnav_baselines.experiments.ithor.objectnav_ithor_base import (
    ObjectNaviThorBaseConfig,
)
from projects.objectnav_detection_based.mixins import \
    ResNetGroundingDINOPreprocessGRUActorCriticMixin
from projects.objectnav_segmentation_based.mixins import (
    ObjectNavPPOMixin,
)


class ObjectNaviThorRGBPPOExperimentConfig(ObjectNaviThorBaseConfig):
    """An Object Navigation experiment configuration in iThor with RGB
    input."""

    SENSORS = [
        RGBSensorThor(
            height=ObjectNaviThorBaseConfig.SCREEN_SIZE,
            width=ObjectNaviThorBaseConfig.SCREEN_SIZE,
            use_resnet_normalization=False,
            uuid="rgb_lowres",
        ),
        GoalObjectTypeThorSensor(object_types=ObjectNaviThorBaseConfig.TARGET_TYPES,),
    ]

    def __init__(self, **kwargs):
        # filter kwargs for the sake of the superclass
        super_kwargs = {
            k: kwargs[k] for k in kwargs if k in ObjectNaviThorBaseConfig.__init__.__code__.co_varnames
        }
        super().__init__(**super_kwargs)

        self.preprocessing_and_model = ResNetGroundingDINOPreprocessGRUActorCriticMixin(
            target_list=list(self.TARGET_TYPES),
            sensors=self.SENSORS,
            resnet_type="RN18",
            screen_size=self.SCREEN_SIZE,
            goal_sensor_type=GoalObjectTypeThorSensor,
            single_class_detection=True if 'single_class_detection' in kwargs and kwargs['single_class_detection'] else False,
        )

    def training_pipeline(self, **kwargs) -> TrainingPipeline:
        return ObjectNavPPOMixin.training_pipeline(
            auxiliary_uuids=[],
            multiple_beliefs=False,
            advance_scene_rollout_period=self.ADVANCE_SCENE_ROLLOUT_PERIOD,
            num_steps=kwargs['num_steps'] if 'num_steps' in kwargs else 8,
        )

    def preprocessors(self) -> Sequence[Union[Preprocessor, Builder[Preprocessor]]]:
        return self.preprocessing_and_model.preprocessors()

    def create_model(self, **kwargs) -> nn.Module:
        return self.preprocessing_and_model.create_model(
            num_actions=self.ACTION_SPACE.n, **kwargs
        )

    @classmethod
    def tag(cls):
        return "ObjectNav-iTHOR-RGB-ResNet18-GroundingDINO-GRU-DDPPO"
