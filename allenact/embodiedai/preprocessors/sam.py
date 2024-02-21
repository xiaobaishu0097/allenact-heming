from typing import List, Callable, Optional, Any, cast, Dict

import copy
import yaml
import einops
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms
import matplotlib.pyplot as plt

from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.utils.misc_utils import prepare_locals_for_super

from submodules.segment_anything.segment_anything import (
    SamAutomaticMaskGenerator,
    sam_model_registry,
)


TARGETS = [
    "AlarmClock",
    "Apple",
    "BaseballBat",
    "BasketBall",
    "Bowl",
    "GarbageCan",
    "HousePlant",
    "Laptop",
    "Mug",
    "RemoteControl",
    "SprayBottle",
    "Television",
    "Vase",
]


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones(
        (
            sorted_anns[0]["segmentation"].shape[0],
            sorted_anns[0]["segmentation"].shape[1],
            4,
        )
    )
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


class SAMEmbedder(nn.Module):
    def __init__(self, sam_config: str):
        super().__init__()

        self.sam_config: str = sam_config
        with open(self.sam_config, "r") as file:
            self.config = yaml.safe_load(file)

        self.model = sam_model_registry[self.config["checkpoint"]["model_type"]](
            checkpoint=self.config["checkpoint"]["path"]
        ).cuda()

        if not self.config["fine_tune"]["image_encoder"]:
            for param in self.model.image_encoder.parameters():
                param.requires_grad = False

        if not self.config["fine_tune"]["mask_decoder"]:
            for param in self.model.mask_decoder.parameters():
                param.requires_grad = False

        if (
            not self.config["fine_tune"]["image_encoder"]
            and not self.config["fine_tune"]["mask_decoder"]
        ):
            self.eval()

        self.mask_generator = SamAutomaticMaskGenerator(
            self.model,
            points_per_side=self.config["model"]["points_per_side"],
            points_per_batch=self.config["model"]["points_per_batch"],
        )

    def embed_segmentation_results(self, segmentation_outputs: list) -> dict:
        segmentation_masks: list = []
        segmentation_features: list = []
        segmentation_areas: list = []
        segmentation_bboxs: list = []
        segmentation_stability_scores: list = []

        if len(segmentation_outputs) > 100:
            segmentation_outputs = sorted(
                segmentation_outputs,
                key=lambda x: x["stability_score"],
                reverse=True,
            )[:100]

        for seg in segmentation_outputs:
            segmentation_masks.append(
                copy.deepcopy(einops.rearrange(seg["segmentation"], "h w -> 1 h w"))
            )
            segmentation_features.append(
                copy.deepcopy(
                    einops.rearrange(
                        F.max_pool2d(
                            torch.tensor(seg["segmentation_feature"]), (64, 64)
                        ),
                        "c 1 1 -> 1 c",
                    )
                )
            )
            segmentation_areas.append(seg["area"])
            segmentation_bboxs.append(
                copy.deepcopy(einops.rearrange(torch.tensor(seg["bbox"]), "c -> 1 c"))
            )
            segmentation_stability_scores.append(seg["stability_score"])

        if len(segmentation_masks) == 0:
            return {
                "segmentation_masks": F.pad(
                    torch.tensor(np.zeros((1, 224, 224))),
                    (0, 0, 0, 0, 0, 100 - 1),
                    "constant",
                    0,
                ),
                "segmentation_features": F.pad(
                    torch.tensor(np.zeros((1, 256))),
                    (0, 0, 0, 100 - 1),
                    "constant",
                    0,
                ),
                "segmentation_areas": F.pad(
                    torch.tensor([0]),
                    (0, 100 - 1),
                    "constant",
                    0,
                ),
                "segmentation_bboxs": F.pad(
                    torch.tensor(np.zeros((1, 4))),
                    (0, 0, 0, 100 - 1),
                    "constant",
                    0,
                ),
                "segmentation_stability_scores": F.pad(
                    torch.tensor([0]),
                    (0, 100 - 1),
                    "constant",
                    0,
                ),
            }

        segmentation_results = {
            "segmentation_masks": F.pad(
                torch.tensor(np.concatenate(segmentation_masks, axis=0)),
                (0, 0, 0, 0, 0, 100 - len(segmentation_masks)),
                "constant",
                0,
            ),
            "segmentation_features": F.pad(
                torch.cat(segmentation_features, dim=0),
                (0, 0, 0, 100 - len(segmentation_features)),
                "constant",
                0,
            ),
            "segmentation_areas": F.pad(
                torch.tensor(segmentation_areas),
                (0, 100 - len(segmentation_areas)),
                "constant",
                0,
            ),
            "segmentation_bboxs": F.pad(
                torch.tensor(np.concatenate(segmentation_bboxs, axis=0)),
                (0, 0, 0, 100 - len(segmentation_bboxs)),
                "constant",
                0,
            ),
            "segmentation_stability_scores": F.pad(
                torch.tensor(segmentation_stability_scores),
                (0, 100 - len(segmentation_stability_scores)),
                "constant",
                0,
            ),
        }

        return segmentation_results

    def merge_list(self, outputs: list):
        segmentation_outputs = {}
        for key in outputs[0].keys():
            segmentation_outputs[key] = torch.stack(
                [output[key] for output in outputs], dim=0
            )

        return segmentation_outputs

    @torch.no_grad()
    def forward(self, x):
        # FIXME: check the input format and type convertion
        segmentation_outputs = []
        for i_batch in range(x["rgb_uint8"].shape[0]):
            output = self.mask_generator.predict_and_extract(
                x["rgb_uint8"][i_batch, ...].cpu().numpy()
            )

            segmentation_outputs.append(self.embed_segmentation_results(output))

        segmentation_outputs = self.merge_list(segmentation_outputs)

        return segmentation_outputs


class SAMPreprocessor(Preprocessor):
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
        sam_config: Optional[str] = None,
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

        self.sam_config = (
            sam_config
            if sam_config is not None
            else "./configs/segment_anything/frozen.yaml"
        )
        self._sam: Optional[SAMEmbedder] = None

        low = -np.inf
        high = np.inf
        shape = (self.output_dims, self.output_height, self.output_width)

        assert (
            len(input_uuids) == 1
        ), "resnet preprocessor can only consume one observation type"

        observation_space = gym.spaces.Box(low=low, high=high, shape=shape)

        targets_features_path = "./datasets/ithor-classagnostic/training_target_features/vit_b_mask_decoder_features.pt"
        self.targets_features = torch.load(targets_features_path)

        super().__init__(**prepare_locals_for_super(locals()))

    @property
    def sam(self) -> SAMEmbedder:
        if self._sam is None:
            self._sam = SAMEmbedder(self.sam_config)
        return self._sam

    def to(self, device: torch.device) -> "SAMPreprocessor":
        self.sam.model = self.sam.model.to(device)
        self.device = device
        return self

    def process(self, obs: Dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        segmentation_output = self.sam(obs)

        targets = []
        for target_id in obs["goal_object_type_ind"]:
            target_features = []
            for seg in self.targets_features[TARGETS[target_id].lower()]:
                target_features.append(
                    einops.rearrange(
                        F.max_pool2d(
                            torch.tensor(seg["segmentation_feature"]), (64, 64)
                        ),
                        "c 1 1 -> 1 c",
                    )
                )
            targets.append(
                F.pad(
                    torch.cat(target_features, dim=0),
                    (0, 0, 0, 10 - len(target_features)),
                    "constant",
                    0,
                )
            )

        targets = torch.stack(targets, dim=0)

        segmentation_output["target_features"] = targets

        return segmentation_output
