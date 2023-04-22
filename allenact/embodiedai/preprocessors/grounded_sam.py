import argparse
import sys
from typing import Any, Callable, Dict, List, Optional, cast

import einops
import gym
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.utils.misc_utils import prepare_locals_for_super
from allenact_plugins.foundation_model_plugin import GroundedSAMWrapper

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
        # args.device = device

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

    def preprocess_caption(self, caption: str) -> str:
        result = caption.lower().strip()
        if result.endswith("."):
            return result
        return result + "."

    def forward(self, x) -> torch.Tensor:
        with torch.no_grad():
            # FIXME: check the input format and type convertion
            image = einops.rearrange(x['rgb_lowres'], 'b h w c -> b c h w')

            try:
                #
                conjunction = ', '
                target_classes = [
                    TARGETS[item] for item in x['goal_object_type_ind']
                ]
                semantic_masks = self.grounded_sam_wrapper.generate_target_semantic_mask(
                    image, [
                        self.preprocess_caption(
                            conjunction.join([target_class, 'ground']))
                        for target_class in target_classes
                    ])
            except:
                print('something went wrong')

            return semantic_masks


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