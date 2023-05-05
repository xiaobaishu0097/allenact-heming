from typing import Sequence, Union, Type

import attr
import gym
import torch.nn as nn
from torchvision import models

from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.base_abstractions.sensor import Sensor
from allenact.embodiedai.preprocessors.resnet import ResNetPreprocessor
from allenact.embodiedai.preprocessors.grounded_sam import GroundedSAMPreprocessor
from allenact.embodiedai.sensors.vision_sensors import RGBSensor, DepthSensor
from allenact.utils.experiment_utils import Builder
from allenact_plugins.navigation_plugin.objectnav.groundedsam_models import GroundedSAMTensorNavActorCritic
from allenact_plugins.robothor_plugin.robothor_tasks import ObjectNavTask


@attr.s(kw_only=True)
class ResNet18GroundedSAMPreprocessGRUActorCriticMixin:
    sensors: Sequence[Sensor] = attr.ib()
    resnet_type: str = attr.ib()

    # # TODO: change the resnet_type to the model used in Grounding DINO and Segment-Anything
    # grounding_dino_model_type: str = attr.ib()
    # segment_anything_model_type: str = attr.ib()

    screen_size: int = attr.ib()
    goal_sensor_type: Type[Sensor] = attr.ib()
    target_list: list[str] = attr.ib()

    def preprocessors(
            self) -> Sequence[Union[Preprocessor, Builder[Preprocessor]]]:
        preprocessors = []

        # TODO: rewrite the following code to use the model used in Grounding DINO and Segment-Anything
        if self.resnet_type in ["RN18", "RN34"]:
            output_shape = (512, 7, 7)
        elif self.resnet_type in ["RN50", "RN101", "RN152"]:
            output_shape = (2048, 7, 7)
        else:
            raise NotImplementedError(
                f"`RESNET_TYPE` must be one 'RNx' with x equaling one of"
                f" 18, 34, 50, 101, or 152.")

        rgb_sensor = next(
            (s for s in self.sensors if isinstance(s, RGBSensor)), None)
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
            # TODO: edit the following code to init the GroundedSAM model
            preprocessors.append(
                GroundedSAMPreprocessor(
                    input_height=self.screen_size,
                    input_width=self.screen_size,
                    output_width=output_shape[2],
                    output_height=output_shape[1],
                    output_dims=output_shape[0],
                    pool=False,
                    input_uuids=[rgb_sensor.uuid],
                    output_uuid="rgb_grounded_sam",
                    target_list=self.target_list,
                ))

        depth_sensor = next(
            (s for s in self.sensors if isinstance(s, DepthSensor)), None)
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
                        models, f"resnet{self.resnet_type.replace('RN', '')}"),
                    input_uuids=[depth_sensor.uuid],
                    output_uuid="depth_resnet_imagenet",
                ))

        return preprocessors

    def create_model(self, **kwargs) -> nn.Module:
        has_rgb = any(isinstance(s, RGBSensor) for s in self.sensors)
        has_depth = any(isinstance(s, DepthSensor) for s in self.sensors)
        goal_sensor_uuid = next(
            (s.uuid
             for s in self.sensors if isinstance(s, self.goal_sensor_type)),
            None,
        )

        # TODO: edit the following code to init the GroundedSAMTensorNavActorCritic
        return GroundedSAMTensorNavActorCritic(
            action_space=gym.spaces.Discrete(
                len(ObjectNavTask.class_action_names())),
            observation_space=kwargs["sensor_preprocessor_graph"].
            observation_spaces,
            goal_sensor_uuid=goal_sensor_uuid,
            rgb_resnet_preprocessor_uuid="rgb_resnet_imagenet" if has_rgb else None,
            rgb_grounded_sam_preprocessor_uuid="rgb_grounded_sam"
            if has_rgb else None,
            depth_grounded_sam_preprocessor_uuid="depth_grounded_sam"
            if has_depth else None,
            hidden_size=512,
            goal_dims=32,
            vit_encoder_layers=kwargs['vit_encoder_layers'] if 'vit_encoder_layers' in kwargs else None,
        )



@attr.s(kw_only=True)
class GroundedSAMPreprocessGRUActorCriticMixin:
    sensors: Sequence[Sensor] = attr.ib()

    # # TODO: change the resnet_type to the model used in Grounding DINO and Segment-Anything
    # grounding_dino_model_type: str = attr.ib()
    # segment_anything_model_type: str = attr.ib()

    screen_size: int = attr.ib()
    goal_sensor_type: Type[Sensor] = attr.ib()
    target_list: list[str] = attr.ib()

    def preprocessors(
            self) -> Sequence[Union[Preprocessor, Builder[Preprocessor]]]:
        preprocessors = []

        # TODO: rewrite the following code to use the model used in Grounding DINO and Segment-Anything
        output_shape = (512, 7, 7)
        # if self.resnet_type in ["RN18", "RN34"]:
        #     output_shape = (512, 7, 7)
        # elif self.resnet_type in ["RN50", "RN101", "RN152"]:
        #     output_shape = (2048, 7, 7)
        # else:
        #     raise NotImplementedError(
        #         f"`RESNET_TYPE` must be one 'RNx' with x equaling one of"
        #         f" 18, 34, 50, 101, or 152.")

        rgb_sensor = next(
            (s for s in self.sensors if isinstance(s, RGBSensor)), None)
        if rgb_sensor is not None:
            # TODO: edit the following code to init the GroundedSAM model
            preprocessors.append(
                GroundedSAMPreprocessor(
                    input_height=self.screen_size,
                    input_width=self.screen_size,
                    output_width=output_shape[2],
                    output_height=output_shape[1],
                    output_dims=output_shape[0],
                    pool=False,
                    input_uuids=[rgb_sensor.uuid],
                    output_uuid="rgb_grounded_sam",
                    target_list=self.target_list,
                ))

        depth_sensor = next(
            (s for s in self.sensors if isinstance(s, DepthSensor)), None)
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
                        models, f"resnet{self.resnet_type.replace('RN', '')}"),
                    input_uuids=[depth_sensor.uuid],
                    output_uuid="depth_resnet_imagenet",
                ))

        return preprocessors

    def create_model(self, **kwargs) -> nn.Module:
        has_rgb = any(isinstance(s, RGBSensor) for s in self.sensors)
        has_depth = any(isinstance(s, DepthSensor) for s in self.sensors)
        goal_sensor_uuid = next(
            (s.uuid
             for s in self.sensors if isinstance(s, self.goal_sensor_type)),
            None,
        )

        # TODO: edit the following code to init the GroundedSAMTensorNavActorCritic
        return GroundedSAMTensorNavActorCritic(
            action_space=gym.spaces.Discrete(
                len(ObjectNavTask.class_action_names())),
            observation_space=kwargs["sensor_preprocessor_graph"].
            observation_spaces,
            goal_sensor_uuid=goal_sensor_uuid,
            rgb_grounded_sam_preprocessor_uuid="rgb_grounded_sam"
            if has_rgb else None,
            depth_grounded_sam_preprocessor_uuid="depth_grounded_sam"
            if has_depth else None,
            hidden_size=512,
            goal_dims=32,
            vit_encoder_layers=kwargs['vit_encoder_layers'] if 'vit_encoder_layers' in kwargs else None,
        )

