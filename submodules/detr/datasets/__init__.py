# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision

from .coco import build as build_coco


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)
    if args.dataset_file == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)
    if args.dataset_file == 'iTHOR_Detection_Data':
        from .coco import build_AI2THOR_dataset
        return build_AI2THOR_dataset(image_set, args)
    if args.dataset_file == 'RoboTHOR_Detection_Data':
        from .coco import build_RoboTHOR_dataset
        return build_RoboTHOR_dataset(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
