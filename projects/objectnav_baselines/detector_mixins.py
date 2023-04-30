from typing import Sequence, Union, Optional, Dict, Tuple, Type

import attr
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torchvision import models

from allenact.algorithms.onpolicy_sync.losses import PPO
from allenact.algorithms.onpolicy_sync.losses.abstract_loss import (
    AbstractActorCriticLoss,
)
from allenact.algorithms.onpolicy_sync.losses.imitation import Imitation
from allenact.algorithms.onpolicy_sync.losses.ppo import PPOConfig
from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.base_abstractions.sensor import Sensor
from allenact.embodiedai.aux_losses.losses import (
    InverseDynamicsLoss,
    TemporalDistanceLoss,
    CPCA1Loss,
    CPCA2Loss,
    CPCA4Loss,
    CPCA8Loss,
    CPCA16Loss,
    MultiAuxTaskNegEntropyLoss,
    CPCA1SoftMaxLoss,
    CPCA2SoftMaxLoss,
    CPCA4SoftMaxLoss,
    CPCA8SoftMaxLoss,
    CPCA16SoftMaxLoss,
)
from allenact.embodiedai.preprocessors.resnet import ResNetPreprocessor
from allenact.embodiedai.preprocessors.detr import DETRPreprocessor
from allenact.embodiedai.sensors.vision_sensors import RGBSensor, DepthSensor
from allenact.utils.experiment_utils import (
    Builder,
    TrainingPipeline,
    PipelineStage,
    LinearDecay,
)
from allenact_plugins.ithor_plugin.ithor_sensors import GoalObjectTypeThorSensor
from allenact_plugins.navigation_plugin.objectnav.models import (
    ResnetTensorNavActorCritic,
    ObjectNavActorCritic,
)
from allenact_plugins.navigation_plugin.objectnav.detr_models import ResnetDETRTensorNavActorCritic
from allenact_plugins.robothor_plugin.robothor_tasks import ObjectNavTask


@attr.s(kw_only=True)
class ResNetDETRPreprocessGRUActorCriticMixin:
    sensors: Sequence[Sensor] = attr.ib()
    resnet_type: str = attr.ib()
    screen_size: int = attr.ib()
    goal_sensor_type: Type[Sensor] = attr.ib()

    def preprocessors(self) -> Sequence[Union[Preprocessor, Builder[Preprocessor]]]:
        preprocessors = []

        if self.resnet_type in ["RN18", "RN34"]:
            output_shape = (512, 7, 7)
        elif self.resnet_type in ["RN50", "RN101", "RN152"]:
            output_shape = (2048, 7, 7)
        else:
            raise NotImplementedError(
                f"`RESNET_TYPE` must be one 'RNx' with x equaling one of"
                f" 18, 34, 50, 101, or 152."
            )

        rgb_sensor = next((s for s in self.sensors if isinstance(s, RGBSensor)), None)
        if rgb_sensor is not None:
            preprocessors.append(
                ResNetPreprocessor(
                    input_height=self.screen_size,
                    input_width=self.screen_size,
                    output_width=output_shape[2],
                    output_height=output_shape[1],
                    output_dims=output_shape[0],
                    pool=False,
                    torchvision_resnet_model=getattr(
                        models, f"resnet{self.resnet_type.replace('RN', '')}"
                    ),
                    input_uuids=[rgb_sensor.uuid],
                    output_uuid="rgb_resnet_imagenet",
                )
            )
            preprocessors.append(
                DETRPreprocessor(
                    input_height=self.screen_size,
                    input_width=self.screen_size,
                    output_width=output_shape[2],
                    output_height=output_shape[1],
                    output_dims=output_shape[0],
                    pool=False,
                    torchvision_resnet_model=getattr(
                        models, f"resnet{self.resnet_type.replace('RN', '')}"
                    ),
                    input_uuids=[rgb_sensor.uuid],
                    output_uuid="rgb_detr",
                )
            )

        depth_sensor = next(
            (s for s in self.sensors if isinstance(s, DepthSensor)), None
        )
        if depth_sensor is not None:
            preprocessors.append(
                ResNetPreprocessor(
                    input_height=self.screen_size,
                    input_width=self.screen_size,
                    output_width=output_shape[2],
                    output_height=output_shape[1],
                    output_dims=output_shape[0],
                    pool=False,
                    torchvision_resnet_model=getattr(
                        models, f"resnet{self.resnet_type.replace('RN', '')}"
                    ),
                    input_uuids=[depth_sensor.uuid],
                    output_uuid="depth_resnet_imagenet",
                )
            )

        return preprocessors

    def create_model(self, **kwargs) -> nn.Module:
        has_rgb = any(isinstance(s, RGBSensor) for s in self.sensors)
        has_depth = any(isinstance(s, DepthSensor) for s in self.sensors)
        goal_sensor_uuid = next(
            (s.uuid for s in self.sensors if isinstance(s, self.goal_sensor_type)),
            None,
        )

        return ResnetDETRTensorNavActorCritic(
            action_space=gym.spaces.Discrete(len(ObjectNavTask.class_action_names())),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            goal_sensor_uuid=goal_sensor_uuid,
            rgb_resnet_preprocessor_uuid="rgb_resnet_imagenet" if has_rgb else None,
            rgb_detr_preprocessor_uuid="rgb_detr" if has_rgb else None,
            depth_resnet_preprocessor_uuid="depth_resnet_imagenet"
            if has_depth
            else None,
            hidden_size=512,
            goal_dims=32,
        )

