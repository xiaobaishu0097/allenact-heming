from typing import List, Callable, Optional, Any, cast, Dict

import copy
import yaml
import einops
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.utils.misc_utils import prepare_locals_for_super

from submodules.EfficientSAM.efficient_sam.build_efficient_sam import (
    build_efficient_sam_vitt,
    build_efficient_sam_vits,
)
from submodules.segment_anything.segment_anything.utils.amg import (
    calculate_stability_score,
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


class EfficientSAMEmbedder(nn.Module):
    def __init__(self, efficient_sam_config: str):
        super().__init__()

        self.efficient_sam_config: str = efficient_sam_config
        with open(self.efficient_sam_config, "r") as file:
            self.config = yaml.safe_load(file)

        if self.config["model"]["type"] == "efficient_sam_vitt":
            self.model = build_efficient_sam_vitt()
        elif self.config["model"]["type"] == "efficient_sam_vits":
            self.model = build_efficient_sam_vits()
        else:
            raise NotImplementedError(
                f"Model type {self.config['model']['type']} not implemented"
            )

        for param in self.model.image_encoder.parameters():
            param.requires_grad = False

        for param in self.model.mask_decoder.parameters():
            param.requires_grad = False

        self.eval()

        xy = []
        self.grid_size = self.config["model"]["grid_size"]
        for i in range(self.grid_size):
            curr_x = 0.5 + i / self.grid_size * 224
            for j in range(self.grid_size):
                curr_y = 0.5 + j / self.grid_size * 224
                xy.append([curr_x, curr_y])
        xy = torch.from_numpy(np.array(xy))

        self.points = torch.from_numpy(np.array(xy)).cuda()
        self.num_pts = xy.shape[0]
        self.point_labels = torch.ones(self.num_pts, 1).cuda()

        self.num_queries = self.config["model"]["num_queries"]

        self.mask_feature_pooling = nn.MaxPool2d((64, 64))

    def get_predictions_given_embeddings_and_queries(self, img):
        predicted_masks, predicted_iou, extracted_features = self.model.extract_masks(
            img.float() / 255,
            einops.repeat(self.points, "n c -> b n 1 c", b=img.shape[0]),
            einops.repeat(self.point_labels, "n c -> b n c", b=img.shape[0]),
        )
        sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
        predicted_iou_scores = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
        predicted_masks = torch.take_along_dim(
            predicted_masks, sorted_ids[..., None, None], dim=2
        )

        batch_masks = []
        batch_iou_ = []
        batch_features = []

        for i_sample in range(predicted_masks.shape[0]):
            sample_predicted_masks = predicted_masks[i_sample]
            sample_iou = predicted_iou_scores[i_sample, :, 0]
            sample_extracted_features = extracted_features[i_sample, ...]

            index_iou = sample_iou > 0.7
            sample_iou_ = sample_iou[index_iou]
            sample_masks = sample_predicted_masks[index_iou]
            sample_features = sample_extracted_features[index_iou]
            sample_score = calculate_stability_score(sample_masks, 0.0, 1.0)
            sample_score = sample_score[:, 0]
            sample_index = sample_score > 0.9

            sample_score_ = sample_score[sample_index]
            sample_masks = sample_masks[sample_index]
            sample_iou_ = sample_iou_[sample_index]
            sample_features = sample_features[sample_index]
            sample_features = einops.rearrange(
                self.mask_feature_pooling(sample_features), "n c 1 1 -> n c"
            )

            if sample_masks.shape[0] >= self.num_queries:
                sorted_indices = torch.argsort(sample_score_, descending=True)
                sample_masks = sample_masks[sorted_indices[: self.num_queries]]
                sample_iou_ = sample_iou_[sorted_indices[: self.num_queries]]
                sample_features = sample_features[sorted_indices[: self.num_queries]]
            else:
                sample_masks = F.pad(
                    sample_masks,
                    (0, 0, 0, 0, 0, 0, 0, self.num_queries - sample_masks.shape[0]),
                    "constant",
                    0,
                )
                sample_iou_ = F.pad(
                    sample_iou_,
                    (0, self.num_queries - sample_iou_.shape[0]),
                    "constant",
                    0,
                )
                sample_features = F.pad(
                    sample_features,
                    (0, 0, 0, self.num_queries - sample_features.shape[0]),
                    "constant",
                    0,
                )

            batch_masks.append(sample_masks)
            batch_iou_.append(sample_iou_)
            batch_features.append(sample_features)

        masks = torch.stack(batch_masks, dim=0)
        iou_ = torch.stack(batch_iou_, dim=0)
        extracted_features = torch.stack(batch_features, dim=0)

        masks = torch.ge(masks, 0.0)
        return masks, iou_, extracted_features

    @torch.no_grad()
    def forward(self, x):
        # FIXME: check the input format and type convertion
        predicted_masks, predicted_iou, extracted_features = (
            self.get_predictions_given_embeddings_and_queries(
                einops.rearrange(x["rgb_uint8"], "b h w c -> b c h w")
            )
        )

        segmentation_outputs = {
            "segmentation_masks": predicted_masks,
            "segmentation_iou": predicted_iou,
            "segmentation_features": extracted_features,
        }

        return segmentation_outputs


class EfficientSAMPreprocessor(Preprocessor):
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
        efficient_sam_config: Optional[str] = None,
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

        self.efficient_sam_config = (
            efficient_sam_config
            if efficient_sam_config is not None
            else "./configs/segment_anything/frozen.yaml"
        )
        self._efficient_sam: Optional[EfficientSAMEmbedder] = None

        low = -np.inf
        high = np.inf
        shape = (self.output_dims, self.output_height, self.output_width)

        assert (
            len(input_uuids) == 1
        ), "resnet preprocessor can only consume one observation type"

        observation_space = gym.spaces.Box(low=low, high=high, shape=shape)

        targets_features_path = "./datasets/ithor-classagnostic/training_target_features/efficientsam_vit_ti_mask_decoder_features.pt"
        self.targets_features = torch.load(targets_features_path)

        super().__init__(**prepare_locals_for_super(locals()))

    @property
    def efficient_sam(self) -> EfficientSAMEmbedder:
        if self._efficient_sam is None:
            self._efficient_sam = EfficientSAMEmbedder(self.efficient_sam_config)
        return self._efficient_sam

    def to(self, device: torch.device) -> "EfficientSAMPreprocessor":
        self.efficient_sam.model = self.efficient_sam.model.to(device)
        self.device = device
        return self

    def process(self, obs: Dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        segmentation_output = self.efficient_sam(obs)

        targets = []
        for target_id in obs["goal_object_type_ind"]:
            target_features = self.targets_features[TARGETS[target_id].lower()][
                "segmentation_features"
            ]
            targets.append(
                F.pad(
                    torch.tensor(target_features),
                    (0, 0, 0, 15 - len(target_features)),
                    "constant",
                    0,
                )
            )

        targets = torch.stack(targets, dim=0)

        segmentation_output["target_features"] = targets

        return segmentation_output
