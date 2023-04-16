from typing import List, Callable, Optional, Any, cast, Dict

import os
import argparse
import einops
import sys
import gym
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms

from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.utils.misc_utils import prepare_locals_for_super

from allenact_plugins.foundation_model_plugin import GroundedSAMWrapper

from submodules.detr.apis.inference import (inference_detector, init_detector,
                                            make_detr_transforms)
from submodules.detr.util.parse import get_args_parser

TARGETS = [
    'AlarmClock',
    'Apple',
    'BaseballBat',
    'BasketBall',
    'Bowl',
    'GarbageCan',
    'HousePlant',
    'Laptop',
    'Mug',
    'RemoteControl',
    'SprayBottle',
    'Television',
    'Vase',
]


def get_grounded_sam_args_parser():
    parser = argparse.ArgumentParser('Set Grounded Segment-Anything',
                                     add_help=False)
    parser.add_argument(
        "--config",
        type=str,
        default=
        './submodules/grounded_sam/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py',
        help="path to config file")
    parser.add_argument(
        "--grounding_dino_checkpoint",
        type=str,
        default=
        './allenact_plugins/foundation_model_plugin/checkpoints/groundingdino_swint_ogc.pth',
        help="path to checkpoint file")
    parser.add_argument(
        "--sam_checkpoint",
        type=str,
        default=
        './allenact_plugins/foundation_model_plugin/checkpoints/sam_vit_h_4b8939.pth',
        help="path to checkpoint file")

    parser.add_argument("--box_threshold",
                        type=float,
                        default=0.3,
                        help="box threshold")
    parser.add_argument("--text_threshold",
                        type=float,
                        default=0.25,
                        help="text threshold")

    return parser


# TODO: edit this to use the new grounded SAM
class GroundedSAMEmbedder(nn.Module):

    def __init__(self, device: torch.device):
        super().__init__()

        argv = sys.argv
        sys.argv = []
        args = get_grounded_sam_args_parser().parse_args()
        args.device = device

        # args.grounding_dino_checkpoint = os.path.join(os.getcwd(), args.grounding_dino_checkpoint)
        # args.sam_checkpoint = os.path.join(os.getcwd(), args.sam_checkpoint)

        sys.argv = argv

        self.transform = transforms.Resize(800)

        self.grounded_sam_wrapper = GroundedSAMWrapper(vars(args))

        self.grounded_sam_wrapper.frozen_parameters()
        self.image_size = nn.Parameter(torch.tensor([[224, 224]]),
                                       requires_grad=False)

        self.eval()

    @property
    def grounding_dino_model(self):
        return self.grounded_sam_wrapper.grounding_dino_model

    @property
    def segment_anything_model(self):
        return self.grounded_sam_wrapper.segment_anything_model

    def embed_detection_results(self, scores, labels, boxes, output, target):
        """
        score: (batch_size, num_detections)
        labels: (batch_size, num_detections)
        boxes: (batch_size, num_detections, 4)
        """
        current_detection_features = torch.cat((
            output['encoder_features'],
            scores,
            labels,
            boxes,
        ),
                                               dim=-1)

        sorted_labels, sort_index = torch.sort(labels, dim=1)

        current_detection_features = torch.gather(
            current_detection_features, 1,
            sort_index.expand_as(current_detection_features))
        current_detection_features[(sorted_labels == (
            len(TARGETS) + 1)).expand_as(current_detection_features)] = 0

        detection_inputs = {
            'features': current_detection_features[..., :256],
            'scores': current_detection_features[..., 256, None],
            'labels': current_detection_features[..., 257, None],
            'bboxes': current_detection_features[..., -4:],
            'target': target,
        }

        # generate target indicator array based on detection results labels
        detection_inputs['indicator'] = (
            detection_inputs['labels'] == einops.repeat(
                target, 'b -> b n 1',
                n=100).expand_as(detection_inputs['labels'])).float()

        return detection_inputs

    def forward(self, x):
        with torch.no_grad():
            # FIXME: check the input format and type convertion
            image = einops.rearrange(x['rgb_lowres'], 'b h w c -> b c h w')

            # TODO: input the RGB image and output the segmenation maps
            output = inference_detector(self.detector, self.transform(image),
                                        None)
            result = self.postprocessor['bbox'](
                output,
                einops.repeat(self.image_size,
                              '1 s -> b s',
                              b=output['pred_logits'].shape[0]))

            pred_scores = torch.cat(
                [einops.rearrange(x['scores'], 'l -> 1 l 1') for x in result],
                dim=0)
            pred_labels = torch.cat(
                [einops.rearrange(x['labels'], 'l -> 1 l 1') for x in result],
                dim=0)
            pred_boxes = torch.cat(
                [einops.rearrange(x['boxes'], 'l b -> 1 l b') for x in result],
                dim=0)

            detection_outputs = self.embed_detection_results(
                pred_scores, pred_labels, pred_boxes, output,
                x['goal_object_type_ind'])

            return detection_outputs


class GroundedSAMPreprocessor(Preprocessor):
    """Preprocess RGB or depth image using a ResNet model."""

    def __init__(
        self,
        input_uuids: List[str],
        output_uuid: str,
        input_height: int,
        input_width: int,
        output_height: int,
        output_width: int,
        output_dims: int,
        pool: bool,
        device: Optional[torch.device] = None,
        device_ids: Optional[List[torch.device]] = None,
        **kwargs: Any,
    ):
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.output_dims = output_dims
        self.pool = pool

        self.device = torch.device("cpu") if device is None else device
        self.device_ids = device_ids or cast(
            List[torch.device], list(range(torch.cuda.device_count())))

        self._grounded_sam: Optional[GroundedSAMEmbedder] = None

        low = -np.inf
        high = np.inf
        shape = (self.output_dims, self.output_height, self.output_width)

        assert (len(input_uuids) == 1
                ), "resnet preprocessor can only consume one observation type"

        observation_space = gym.spaces.Box(low=low, high=high, shape=shape)

        super().__init__(**prepare_locals_for_super(locals()))

    @property
    def grounded_sam(self) -> GroundedSAMEmbedder:
        if self._grounded_sam is None:
            self._grounded_sam = GroundedSAMEmbedder(device=self.device)
        return self._grounded_sam

    def to(self, device: torch.device) -> "GroundedSAMPreprocessor":
        self._grounded_sam.grounding_dino_model = self.grounded_sam.grounding_dino_model.to(
            device)
        self._grounded_sam.segment_anything_model.model = self.grounded_sam.segment_anything_model.model.to(
            device)
        self.device = device
        return self

    def process(self, obs: Dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        segmentation_output = self.grounded_sam(obs)
        return segmentation_output