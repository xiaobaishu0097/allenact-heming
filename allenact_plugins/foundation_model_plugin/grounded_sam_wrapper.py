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
from submodules.grounded_sam.GroundingDINO.groundingdino.models import \
    build_model
from submodules.grounded_sam.GroundingDINO.groundingdino.util.box_ops import \
    box_cxcywh_to_xyxy
from submodules.grounded_sam.GroundingDINO.groundingdino.util.slconfig import \
    SLConfig
from submodules.grounded_sam.GroundingDINO.groundingdino.util.utils import (
    clean_state_dict, get_phrases_from_posmap)


class GroundedSAMWrapper:

    def __init__(self, configs: dict) -> None:
        self.logger = get_logger()

        self.device = configs['device'] if 'device' in configs else 'cpu'

        self.grounding_dino_checkpoint = configs[
            'grounding_dino_checkpoint'] if 'grounding_dino_checkpoint' in configs else './checkpoints/groundingdino_swint_ogc.pth'
        self.sam_checkpoint = configs[
            'sam_checkpoint'] if 'sam_checkpoint' in configs else './checkpoints/sam_vit_h_4b8939.pth'

        self.box_threshold = configs[
            'box_threshold'] if 'box_threshold' in configs else 0.3
        self.text_threshold = configs[
            'text_threshold'] if 'text_threshold' in configs else 0.25

        self.dino_config_path = configs[
            'dino_config_path'] if 'dino_config_path' in configs else './submodules/grounded_sam/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'

        self.init_grounded_sam_models()

    def init_grounded_sam_models(self) -> None:
        self.grounding_dino_model
        self.segment_anything_model

    def frozen_parameters(self) -> None:
        for param in self.grounding_dino_model.parameters():
            param.requires_grad = False
        for param in self.segment_anything_model.model.parameters():
            param.requires_grad = False

    def load_grounding_dino_model(self):
        args = SLConfig.fromfile(self.dino_config_path)
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

    def load_image_dino(self, image_path):
        # load image
        image_pil = Image.open(image_path).convert("RGB")  # load image

        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image, _ = transform(image_pil, None)  # 3, h, w
        return image_pil, image

    def check_device_attr(self):
        if self.device != self.segment_anything_model.device:
            self.device = self.segment_anything_model.device

    @property
    def image_mean(self) -> torch.Tensor:
        if not hasattr(self, '_image_mean'):
            self.check_device_attr()
            self._image_mean = torch.tensor([0.485, 0.456, 0.406],
                                            dtype=torch.float32,
                                            device=self.device)
        return self._image_mean

    @property
    def image_std(self) -> torch.Tensor:
        if not hasattr(self, '_image_std'):
            self.check_device_attr()
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
            caption: str) -> dict:
        prediction_logits = pred_logits.sigmoid(
        )  # prediction_logits.shape = (batch_size, nq, 256)
        prediction_boxes = pred_boxes  # prediction_boxes.shape = (batch_size, nq, 4)

        mask = prediction_logits.max(dim=-1)[0] > self.box_threshold
        logits = prediction_logits[mask]  # logits.shape = (batch_size, n, 256)
        boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)

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
            'phrases': phrases
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
                    outputs['pred_boxes'][batch_index], captions[batch_index])
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

    def combine_masks(self, pred_phrases: list,
                      sam_proposal_masks: torch.Tensor,
                      target: list) -> list[torch.Tensor]:
        # combine the masks shared the same phrase
        # return a list of masks with the same order of the targets
        # sam_proposal_masks are torch.Tensor on the cuda device
        masks = []
        for t in target:
            if t not in pred_phrases:
                mask = torch.zeros_like(sam_proposal_masks[0],
                                        device=self.device)
            else:
                matching_indices = torch.tensor([
                    index for index, pred_phrase in enumerate(pred_phrases)
                    if pred_phrase == t
                ]).to(self.device)
                if matching_indices.shape[0] > 1:
                    mask = self.element_wise_logical_or(
                        sam_proposal_masks.index_select(0, matching_indices))
                else:
                    mask = sam_proposal_masks[matching_indices[0]]
            masks.append(mask)
        return masks

    def element_wise_logical_or(self, masks: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(masks[0, ...], dtype=torch.bool)
        for mask_index in range(masks.shape[0]):
            result = torch.logical_or(result, masks[mask_index, ...])
        return result

    def generate_target_semantic_mask(
            self,
            image: torch.Tensor,  # image.shape = (n, 3, h, w)
            captions: list[str],  # captions length = n
    ) -> torch.Tensor:
        # target should be a list including the target phrases, like ['cup', 'floor']
        # targets will be concatenated with conjunction into a single string as caption for the GroundingDINO model
        assert image.shape[0] == len(
            captions
        ), 'The number of targets should be the same as the number of images'

        batch_boxes_phrases = self.get_grounding_dino_output(image, captions)

        with torch.cuda.device(self.device):
            batch_sam_proposal_masks = []
            for batch_index in range(image.shape[0]):
                sam_proposal_masks = self.generate_object_sam_proposal_masks(
                    image[batch_index, ...],
                    batch_boxes_phrases[batch_index]['boxes'])
                # based on the predicted phrases, we combine the masks shared the same phrase,
                # and return a list of masks with the same order of the targets

                sam_proposal_masks = self.combine_masks(
                    batch_boxes_phrases[batch_index]['phrases'],
                    sam_proposal_masks['masks'], captions[batch_index].replace(' ', '').replace('.', '').split(','))

                batch_sam_proposal_masks.append(torch.cat(sam_proposal_masks, dim=0))

        return torch.stack(batch_sam_proposal_masks)

    def generate_grounding_dino_bounding_boxes(self, image_path: str,
                                               text_prompt: str) -> dict:
        image_pil, image = self.load_image_dino(image_path)
        boxes_filt, _, pred_phrases = self.get_grounding_dino_output(
            torch.tensor(image), text_prompt)
        return {'boxes': boxes_filt, 'pred_phrases': pred_phrases}

    @property
    def segment_anything_model(self):
        if not hasattr(self, '_segment_anything_model'):
            self._segment_anything_model = SamPredictor(
                build_sam(checkpoint=self.sam_checkpoint).to(self.device))

            # if self.device != self.segment_anything_model.device:
            # self.device = self.segment_anything_model.device

        return self._segment_anything_model

    # @property
    # def device(self):
    #     return self.segment_anything_model.device

    def load_image_sam(self, image_path: str):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    @torch.no_grad()
    def generate_object_sam_proposal_masks(self, image: torch.Tensor,
                                           boxes_filt: torch.Tensor) -> dict:
        with torch.cuda.device(self.device):
            # TODO: avoid the conversion between numpy and torch.Tensor
            image = (einops.rearrange(image, 'c h w -> h w c').cpu().numpy() *
                     255).astype(np.uint8)
            self.segment_anything_model.set_image(image)

            H, W = image.shape[:2]
            boxes_filt = box_cxcywh_to_xyxy(boxes_filt) * torch.Tensor(
                [W, H, W, H]).to(self.segment_anything_model.device)
            # for i in range(boxes_filt.size(0)):
            #     boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H]).to(self.segment_anything_model.device)
            #     boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            #     boxes_filt[i][2:] += boxes_filt[i][:2]

            # boxes_filt = boxes_filt
            transformed_boxes = self.segment_anything_model.transform.apply_boxes_torch(
                boxes_filt, (H, W))

            if transformed_boxes.shape[0] > 0:
                masks, _, _ = self.segment_anything_model.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False,
                )
            else:
                masks = torch.zeros((1, 1, H, W),
                                    dtype=torch.bool,
                                    device=self.device)

            return {'masks': masks}



if __name__ == '__main__':
    pass