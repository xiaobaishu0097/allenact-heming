from typing import List, Callable, Optional, Any, cast, Dict

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
    'AlarmClock', 'Apple', 'BaseballBat', 'BasketBall', 'Bowl', 'GarbageCan', 'HousePlant',
    'Laptop', 'Mug', 'RemoteControl', 'SprayBottle', 'Television', 'Vase',
]

# TODO: edit this to use the new grounded SAM
class GroundedSAMEmbedder(nn.Module):
    def __init__(self, grounding_dino_model_path: Optional[str] = None):
        super().__init__()

        argv = sys.argv
        sys.argv = []
        args = get_args_parser().parse_args()
        sys.argv = argv

        self.transform = transforms.Resize(800)

        self.grounded_sam_wrapper = GroundedSAMWrapper(vars(args))

        if grounding_dino_model_path is None:
            grounding_dino_model_path = './storage/model_zoo/detr/robothor.13cls.checkpoint0059.pth'

        self.grounded_sam_wrapper.frozen_parameters()

        self.image_size = nn.Parameter(torch.tensor([[224, 224]]), requires_grad=False)

        self.eval()

    def embed_detection_results(self, scores, labels, boxes, output, target):
        """
        score: (batch_size, num_detections)
        labels: (batch_size, num_detections)
        boxes: (batch_size, num_detections, 4)
        """
        current_detection_features = torch.cat(
            (output['encoder_features'], scores, labels, boxes,), dim=-1)

        sorted_labels, sort_index = torch.sort(labels, dim=1)

        current_detection_features = torch.gather(current_detection_features, 1, sort_index.expand_as(current_detection_features))
        current_detection_features[(sorted_labels==(len(TARGETS)+1)).expand_as(current_detection_features)] = 0

        detection_inputs = {
            'features': current_detection_features[..., :256],
            'scores': current_detection_features[..., 256, None],
            'labels': current_detection_features[..., 257, None],
            'bboxes': current_detection_features[..., -4:],
            'target': target,
        }

        # generate target indicator array based on detection results labels
        detection_inputs['indicator'] = (detection_inputs['labels'] == einops.repeat(target, 'b -> b n 1', n=100).expand_as(detection_inputs['labels'])).float()

        return detection_inputs

    def forward(self, x):
        with torch.no_grad():
            # FIXME: check the input format and type convertion
            image = einops.rearrange(x['rgb_lowres'], 'b h w c -> b c h w')
            output = inference_detector(self.detector, self.transform(image), None)
            result = self.postprocessor['bbox'](output, einops.repeat(self.image_size, '1 s -> b s', b=output['pred_logits'].shape[0]))

            pred_scores = torch.cat([einops.rearrange(x['scores'], 'l -> 1 l 1') for x in result], dim=0)
            pred_labels = torch.cat([einops.rearrange(x['labels'], 'l -> 1 l 1') for x in result], dim=0)
            pred_boxes = torch.cat([einops.rearrange(x['boxes'], 'l b -> 1 l b') for x in result], dim=0)

            detection_outputs = self.embed_detection_results(pred_scores, pred_labels, pred_boxes, output, x['goal_object_type_ind'])

            return detection_outputs



class DETRPreprocessor(Preprocessor):
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
        torchvision_resnet_model: Callable[..., models.ResNet] = models.resnet18,
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
            List[torch.device], list(range(torch.cuda.device_count()))
        )

        self._detr: Optional[DETREmbedder] = None

        low = -np.inf
        high = np.inf
        shape = (self.output_dims, self.output_height, self.output_width)

        assert (
            len(input_uuids) == 1
        ), "resnet preprocessor can only consume one observation type"

        observation_space = gym.spaces.Box(low=low, high=high, shape=shape)

        super().__init__(**prepare_locals_for_super(locals()))

    @property
    def detr(self) -> DETREmbedder:
        if self._detr is None:
            self._detr = DETREmbedder()
        return self._detr

    def to(self, device: torch.device) -> "DETRPreprocessor":
        self._detr = self.detr.to(device)
        self.device = device
        return self

    def process(self, obs: Dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        detection_output = self.detr(obs)
        return detection_output