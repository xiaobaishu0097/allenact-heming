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
from allenact_plugins.foundation_model_plugin import GroundingDINOWrapper


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
class GroundingDINOEmbedder(nn.Module):

    def __init__(self,
                 target_list: list[str],
                 device: torch.device,
                 single_class_detection: bool = False):
        super().__init__()

        argv = sys.argv
        sys.argv = []
        args = get_grounded_sam_args_parser().parse_args()
        # args.device = device

        # args.grounding_dino_checkpoint = os.path.join(os.getcwd(), args.grounding_dino_checkpoint)
        # args.sam_checkpoint = os.path.join(os.getcwd(), args.sam_checkpoint)

        sys.argv = argv

        self.transform = transforms.Resize(800)

        self.grounding_dino_wrapper = GroundingDINOWrapper(vars(args))
        self.target_list = target_list
        self.target_list_lower = [item.lower() for item in self.target_list]

        self.grounding_dino_wrapper.frozen_parameters()
        self.image_size = nn.Parameter(torch.tensor([[224, 224]]),
                                       requires_grad=False)

        self.single_class_detection = single_class_detection
        if single_class_detection:
            self.embed_detection_results = self.embed_single_class_detection_results
        else:
            self.embed_detection_results = self.embed_multi_class_detection_results

        self.eval()

    @property
    def grounding_dino_model(self):
        return self.grounding_dino_wrapper.grounding_dino_model

    def preprocess_caption(self, caption: str) -> str:
        result = caption.lower().strip()
        if result.endswith("."):
            return result
        return result + "."

    def embed_single_class_detection_results(self, pred_scores: torch.Tensor,
                                             pred_boxes: torch.Tensor,
                                             pred_features: torch.Tensor,
                                             **kwargs) -> torch.Tensor:
        detection_results = torch.zeros(
            (1, 64, pred_features.shape[-1] + 5),
            dtype=torch.float32,
            device=self.grounding_dino_wrapper.device)

        detection_results[:, :pred_features.shape[0], :pred_features.
                          shape[-1]] = pred_features
        detection_results[:, :pred_features.shape[0],
                          pred_features.shape[-1]] = pred_scores
        detection_results[:, :pred_features.shape[0], -4:] = pred_boxes

        return detection_results

    def embed_multi_class_detection_results(self, pred_scores: torch.Tensor,
                                            pred_labels: torch.Tensor,
                                            pred_boxes: torch.Tensor,
                                            pred_features: torch.Tensor,
                                            target_class,
                                            **kwargs) -> torch.Tensor:
        
        detection_results = torch.zeros(
            (1, 64, pred_features.shape[-1] + 7),
            dtype=torch.float32,
            device=self.grounding_dino_wrapper.device)

        detection_results[:, :pred_features.shape[0], :pred_features.
                          shape[-1]] = pred_features
        detection_results[:, :pred_features.shape[0],
                          pred_features.shape[-1]] = pred_labels
        detection_results[:, :pred_features.shape[0],
                          pred_features.shape[-1]+1] = pred_scores
        detection_results[:, :pred_features.shape[0], -5:-1] = pred_boxes
        detection_results[:, :pred_features.shape[0], -1] = target_class

        return detection_results

    def forward(self, x) -> torch.Tensor:
        with torch.no_grad():
            # FIXME: check the input format and type convertion
            image = einops.rearrange(x['rgb_lowres'], 'b h w c -> b c h w')

            conjunction = ', '
            target_classes = [
                self.target_list[item] for item in x['goal_object_type_ind']
            ]
            grounding_dino_outputs = self.grounding_dino_wrapper.get_grounding_dino_output(
                image, [
                    self.preprocess_caption(conjunction.join([target_class]))
                    for target_class in target_classes
                ])

            grounding_dino_outputs = torch.stack([
                self.embed_detection_results(
                    **{
                        'pred_scores':
                        detection_info['logits'],
                        'pred_labels':
                        torch.tensor(
                            [
                                self.target_list_lower.index(target)
                                for target in detection_info['phrases']
                            ],
                            device=self.grounding_dino_wrapper.device),
                        'pred_boxes':
                        detection_info['boxes'],
                        'pred_features':
                        detection_info['features'],
                        'target_class':
                        x['goal_object_type_ind']
                    }) for detection_info in grounding_dino_outputs
            ])
            return grounding_dino_outputs


class GroundingDINOPreprocessor(Preprocessor):
    """Preprocess RGB or depth image using a ResNet model."""

    def __init__(
        self,
        target_list: List[str],
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
        single_class_detection: bool = False,
        **kwargs: Any,
    ):
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.output_dims = output_dims
        self.pool = pool
        self.target_list = target_list

        self.device = torch.device("cpu") if device is None else device
        self.device_ids = device_ids or cast(
            List[torch.device], list(range(torch.cuda.device_count())))

        self._grounding_dino: Optional[GroundingDINOEmbedder] = None

        low = -np.inf
        high = np.inf
        shape = (self.output_dims, self.output_height, self.output_width)

        assert (len(input_uuids) == 1
                ), "resnet preprocessor can only consume one observation type"

        observation_space = gym.spaces.Box(low=low, high=high, shape=shape)

        self.single_class_detection = single_class_detection

        super().__init__(**prepare_locals_for_super(locals()))

    @property
    def grounding_dino(self) -> GroundingDINOEmbedder:
        if self._grounding_dino is None:
            self._grounding_dino = GroundingDINOEmbedder(
                target_list=self.target_list,
                single_class_detection=self.single_class_detection,
                device=self.device)
        return self._grounding_dino

    def to(self, device: torch.device) -> "GroundingDINOPreprocessor":
        self._grounding_dino.grounding_dino_model = self.grounding_dino.grounding_dino_model.to(
            device)
        self._grounding_dino.grounding_dino_wrapper.device = device
        self.device = device
        return self

    def process(self, obs: Dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        detection_output = self.grounding_dino(obs)
        return detection_output