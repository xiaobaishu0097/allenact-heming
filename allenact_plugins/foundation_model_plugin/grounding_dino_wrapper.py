import copy
import os
from typing import Tuple

import cv2
import einops
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from segment_anything import SamPredictor, build_sam

import submodules.grounded_sam.GroundingDINO.groundingdino.datasets.transforms as T
from allenact.utils.system import get_logger
from allenact_plugins.foundation_model_plugin.groundingdino_feature_extractor import build_model
from submodules.grounded_sam.GroundingDINO.groundingdino.util.box_ops import \
    box_cxcywh_to_xyxy
from submodules.grounded_sam.GroundingDINO.groundingdino.util.slconfig import \
    SLConfig
from submodules.grounded_sam.GroundingDINO.groundingdino.util.utils import (
    clean_state_dict, get_phrases_from_posmap)


class GroundingDINOWrapper:

    def __init__(self, configs: dict) -> None:
        self.logger = get_logger()

        self.device = configs['device'] if 'device' in configs else 'cpu'

        self.grounding_dino_checkpoint = configs[
            'grounding_dino_checkpoint'] if 'grounding_dino_checkpoint' in configs else './checkpoints/groundingdino_swint_ogc.pth'

        self.box_threshold = configs[
            'box_threshold'] if 'box_threshold' in configs else 0.3
        self.text_threshold = configs[
            'text_threshold'] if 'text_threshold' in configs else 0.25

        self.dino_config_path = configs[
            'dino_config_path'] if 'dino_config_path' in configs else './submodules/grounded_sam/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'

    def frozen_parameters(self) -> None:
        for param in self.grounding_dino_model.parameters():
            param.requires_grad = False

    def load_grounding_dino_model(self):
        args = SLConfig.fromfile(self.dino_config_path)
        args.modelname = 'groundingdino_feature_extractor'
        model = build_model(args)
        checkpoint = torch.load(self.grounding_dino_checkpoint,
                                map_location="cpu")
        model.load_state_dict(clean_state_dict(checkpoint["model"]),
                              strict=False)

        model = model.to(self.device)
        model.eval()
        return model

    @property
    def grounding_dino_model(self):
        if not hasattr(self, '_grounding_dino_model'):
            self._grounding_dino_model = self.load_grounding_dino_model()
        return self._grounding_dino_model

    @property
    def image_mean(self) -> torch.Tensor:
        if not hasattr(self, '_image_mean'):
            self._image_mean = torch.tensor([0.485, 0.456, 0.406],
                                            dtype=torch.float32,
                                            device=self.device)
        return self._image_mean

    @property
    def image_std(self) -> torch.Tensor:
        if not hasattr(self, '_image_std'):
            self._image_std = torch.tensor([0.229, 0.224, 0.225],
                                           dtype=torch.float32,
                                           device=self.device)
        return self._image_std

    def resize_and_normalize(
            self,
            image: torch.Tensor,  # image.shape = (n, 3, h, w)
            new_size: tuple) -> torch.Tensor:
        # Resize the tensor
        resized_tensor = F.interpolate(image,
                                       size=new_size,
                                       mode='bilinear',
                                       align_corners=False)

        # Normalize the tensor
        batch_size = image.shape[0]
        normalized_tensor = (resized_tensor - einops.repeat(
            self.image_mean, 'c -> b c 1 1', b=batch_size)) / einops.repeat(
                self.image_std, 'c -> b c 1 1', b=batch_size)

        return normalized_tensor

    def extract_boxes_and_phrases(
            self,
            pred_logits: torch.Tensor,  # pred_logits.shape = (nq, 256)
            pred_boxes: torch.Tensor,  # pred_boxes.shape = (nq, 4)
            pred_features: torch.Tensor, # pred_features.shape = (nq, 256)
            caption: str) -> dict:
        prediction_logits = pred_logits.sigmoid(
        )  # prediction_logits.shape = (batch_size, nq, 256)
        prediction_boxes = pred_boxes.clone()  # prediction_boxes.shape = (batch_size, nq, 4)
        prediction_features = pred_features.clone() # prediction_features.shape = (batch_size, nq, 256)

        mask = prediction_logits.max(dim=-1)[0] > self.box_threshold
        logits = prediction_logits[mask]  # logits.shape = (batch_size, n, 256)
        boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)
        features = prediction_features[mask]

        tokenizer = self.grounding_dino_model.tokenizer
        tokenized = tokenizer(caption)

        phrases = [
            get_phrases_from_posmap(logit > self.text_threshold, tokenized,
                                    tokenizer).replace('.', '')
            for logit in logits
        ]

        return {
            'boxes': boxes,
            'logits': logits.max(dim=1)[0],
            'phrases': phrases,
            'features': features,
        }

    @torch.no_grad()
    def batch_predict(
        self,
        image: torch.Tensor,
        captions: list[str],
    ) -> list:
        batch_size = image.shape[0]

        with torch.cuda.device(self.device):
            image = image.cuda()
            outputs = self.grounding_dino_model(image, captions=captions)

            batch_boxes_phrases = []
            for batch_index in range(batch_size):
                boxes_phrases = self.extract_boxes_and_phrases(
                    outputs['pred_logits'][batch_index],
                    outputs['pred_boxes'][batch_index],
                    outputs['pred_features'][batch_index],
                    captions[batch_index])
                batch_boxes_phrases.append(boxes_phrases)

        return batch_boxes_phrases

    def get_grounding_dino_output(
        self,
        image: torch.Tensor,
        captions: list[str],
    ) -> list:
        image = self.resize_and_normalize(image, new_size=(800, 800))

        batch_boxes_phrases = self.batch_predict(
            image=image,
            captions=captions,
        )
        return batch_boxes_phrases

if __name__ == '__main__':
    pass