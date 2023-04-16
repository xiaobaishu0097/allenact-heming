import os

import cv2
import torch
from PIL import Image
import numpy as np

from allenact.utils.system import get_logger

from segment_anything import build_sam, SamPredictor

import submodules.grounded_sam.GroundingDINO.groundingdino.datasets.transforms as T
from submodules.grounded_sam.GroundingDINO.groundingdino.models import build_model
from submodules.grounded_sam.GroundingDINO.groundingdino.util.slconfig import SLConfig
from submodules.grounded_sam.GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap


class GroundedSAMWrapper:

    def __init__(self, configs: dict) -> None:
        self.logger = get_logger()

        self.device = configs['device'] if 'device' in configs else 'cuda'

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

    def init_grounded_sam_models(self) -> None:
        self.grounding_dino_model
        self.segment_anything_model

    def frozen_parameters(self) -> None:
        for param in self.grounding_dino_model.parameters():
            param.requires_grad = False
        for param in self.segment_anything_model.parameters():
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
            self.logger.info(
                f'Process {os.getpid()} on {self.device} renew the model')
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

    def get_grounding_dino_output(
        self,
        image: torch.Tensor,
        caption: str,
        with_logits=True,
    ):
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."

        image = image.to(self.device)
        with torch.no_grad():
            outputs = self.grounding_dino_model(image[None],
                                                captions=[caption])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
        logits.shape[0]

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > self.box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        logits_filt.shape[0]

        # get phrase
        tokenlizer = self.grounding_dino_model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > self.text_threshold,
                                                  tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase +
                                    f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)

        return boxes_filt, pred_phrases

    def generate_grounding_dino_bounding_boxes(self, image_path: str,
                                               text_prompt: str) -> dict:
        image_pil, image = self.load_image_dino(image_path)
        boxes_filt, pred_phrases = self.get_grounding_dino_output(
            image, text_prompt)
        return {'boxes': boxes_filt, 'pred_phrases': pred_phrases}

    @property
    def segment_anything_model(self):
        if not hasattr(self, '_segment_anything_model'):
            self._segment_anything_model = SamPredictor(
                build_sam(checkpoint=self.sam_checkpoint).to(self.device))
        return self._segment_anything_model

    def load_image_sam(self, image_path: str):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def generate_object_sam_proposal_masks(self, image_path: str,
                                           boxes_filt: torch.Tensor) -> dict:
        with torch.no_grad():
            image = self.load_image_sam(image_path)
            self.segment_anything_model.set_image(image)

            H, W = image.shape[:2]
            for i in range(boxes_filt.size(0)):
                boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                boxes_filt[i][2:] += boxes_filt[i][:2]

            boxes_filt = boxes_filt.cpu()
            transformed_boxes = self.segment_anything_model.transform.apply_boxes_torch(
                boxes_filt, image.shape[:2]).to(self.device)

            if transformed_boxes.shape[0] > 0:
                masks, _, _ = self.segment_anything_model.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False,
                )
            else:
                masks = torch.zeros((1, 1, 300, 300), dtype=torch.bool)

            return {'masks': masks}

    def generate_object_grounded_sam_proposals(self, image_path: str,
                                               text_prompt: str) -> dict:
        grounding_dino_bounding_boxes = self.generate_grounding_dino_bounding_boxes(
            image_path, text_prompt)
        sam_proposal_masks = self.generate_object_sam_proposal_masks(
            image_path, grounding_dino_bounding_boxes['boxes'])
        return {
            'boxes': grounding_dino_bounding_boxes['boxes'],
            'pred_phrases': grounding_dino_bounding_boxes['pred_phrases'],
            'masks': sam_proposal_masks['masks']
        }


if __name__ == '__main__':
    pass