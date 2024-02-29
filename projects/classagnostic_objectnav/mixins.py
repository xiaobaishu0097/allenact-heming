from typing import Sequence, Union, Optional, Type

import attr
import gym
import torch.nn as nn
from torchvision import models

from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.base_abstractions.sensor import Sensor
from allenact.embodiedai.preprocessors import (
    ResNetPreprocessor,
    SAMPreprocessor,
    EfficientSAMPreprocessor,
)
from allenact.embodiedai.sensors.vision_sensors import RGBSensor, DepthSensor
from allenact.utils.experiment_utils import Builder
from allenact_plugins.ithor_plugin.ithor_sensors import GoalObjectTypeThorSensor
from allenact_plugins.navigation_plugin.objectnav import (
    SAMResnetTensorNavActorCritic,
    ObjectNavActorCritic,
)
from allenact_plugins.robothor_plugin.robothor_tasks import ObjectNavTask


@attr.s(kw_only=True)
class SAMResNetPreprocessGRUActorCriticMixin:
    sensors: Sequence[Sensor] = attr.ib()
    resnet_type: str = attr.ib()
    screen_size: int = attr.ib()
    goal_sensor_type: Type[Sensor] = attr.ib()
    sam_config: str = attr.ib()

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
                SAMPreprocessor(
                    input_height=self.screen_size,
                    input_width=self.screen_size,
                    output_width=output_shape[2],
                    output_height=output_shape[1],
                    output_dims=output_shape[0],
                    pool=False,
                    input_uuids=[rgb_sensor.uuid],
                    output_uuid="rgb_sam",
                    sam_config=self.sam_config,
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

        return SAMResnetTensorNavActorCritic(
            action_space=gym.spaces.Discrete(len(ObjectNavTask.class_action_names())),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            goal_sensor_uuid=goal_sensor_uuid,
            rgb_resnet_preprocessor_uuid="rgb_resnet_imagenet" if has_rgb else None,
            rgb_sam_preprocessor_uuid="rgb_sam" if has_rgb else None,
            depth_resnet_preprocessor_uuid=(
                "depth_resnet_imagenet" if has_depth else None
            ),
            hidden_size=512,
            goal_dims=32,
            **kwargs,
        )


@attr.s(kw_only=True)
class EfficientSAMResNetPreprocessGRUActorCriticMixin:
    sensors: Sequence[Sensor] = attr.ib()
    resnet_type: str = attr.ib()
    screen_size: int = attr.ib()
    goal_sensor_type: Type[Sensor] = attr.ib()
    efficient_sam_config: str = attr.ib()

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
                EfficientSAMPreprocessor(
                    input_height=self.screen_size,
                    input_width=self.screen_size,
                    output_width=output_shape[2],
                    output_height=output_shape[1],
                    output_dims=output_shape[0],
                    pool=False,
                    input_uuids=[rgb_sensor.uuid],
                    output_uuid="rgb_sam",
                    efficient_sam_config=self.efficient_sam_config,
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

        return SAMResnetTensorNavActorCritic(
            action_space=gym.spaces.Discrete(len(ObjectNavTask.class_action_names())),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            goal_sensor_uuid=goal_sensor_uuid,
            rgb_resnet_preprocessor_uuid="rgb_resnet_imagenet" if has_rgb else None,
            rgb_sam_preprocessor_uuid="rgb_sam" if has_rgb else None,
            depth_resnet_preprocessor_uuid=(
                "depth_resnet_imagenet" if has_depth else None
            ),
            hidden_size=512,
            goal_dims=32,
            **kwargs,
        )


@attr.s(kw_only=True)
class ObjectNavUnfrozenResNetWithGRUActorCriticMixin:
    backbone: str = attr.ib()
    sensors: Sequence[Sensor] = attr.ib()
    auxiliary_uuids: Sequence[str] = attr.ib()
    add_prev_actions: bool = attr.ib()
    multiple_beliefs: bool = attr.ib()
    belief_fusion: Optional[str] = attr.ib()

    def create_model(self, **kwargs) -> nn.Module:
        rgb_uuid = next(
            (s.uuid for s in self.sensors if isinstance(s, RGBSensor)), None
        )
        depth_uuid = next(
            (s.uuid for s in self.sensors if isinstance(s, DepthSensor)), None
        )
        goal_sensor_uuid = next(
            (s.uuid for s in self.sensors if isinstance(s, GoalObjectTypeThorSensor))
        )

        return ObjectNavActorCritic(
            action_space=gym.spaces.Discrete(len(ObjectNavTask.class_action_names())),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            rgb_uuid=rgb_uuid,
            depth_uuid=depth_uuid,
            goal_sensor_uuid=goal_sensor_uuid,
            hidden_size=(
                192 if self.multiple_beliefs and len(self.auxiliary_uuids) > 1 else 512
            ),
            backbone=self.backbone,
            resnet_baseplanes=32,
            object_type_embedding_dim=32,
            num_rnn_layers=1,
            rnn_type="GRU",
            add_prev_actions=self.add_prev_actions,
            action_embed_size=6,
            auxiliary_uuids=self.auxiliary_uuids,
            multiple_beliefs=self.multiple_beliefs,
            beliefs_fusion=self.belief_fusion,
        )
