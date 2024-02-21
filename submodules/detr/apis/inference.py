import torch

import sys
sys.path.append('submodules/detr')

from models import build_model
import torchvision.transforms as transforms


def init_detector(config, checkpoint=None, device='cuda:0', cfg_options=None):
    model, _, postprocessors = build_model(config)

    model.to(device)
    model.eval()

    return model, postprocessors


def inference_detector(model, images, preprocessors=None):
    if isinstance(images, (list, tuple, torch.Tensor)):
        is_batch = True
    else:
        images = [images]
        is_batch = False

    # device = next(model.parameters()).device

    if preprocessors is not None:
        images = preprocessors(images)

    # forward the model
    with torch.no_grad():
        results = model(images, feature_extraction=True)

    return results
    # if not is_batch:
    #     return results[0]
    # else:
    #     return results


def make_detr_transforms(image_set):

    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomSelect(
                transforms.RandomResize(scales, max_size=1333),
                transforms.Compose([
                    transforms.Resize([400, 500, 600]),
                    transforms.RandomSizeCrop(384, 600),
                    transforms.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([800], max_size=1333),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    if image_set == 'all':
        return transforms.Compose([
            transforms.Resize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


if __name__ == '__main__':
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    import sys
    sys.path.append('submodules/detr')
    from util.parse import get_args_parser

    args = get_args_parser().parse_args()
    args.dataset_file = "RoboTHOR_Detection_Data"

    model, postprocessors = init_detector(args)

    img = cv2.imread('/home/hemingdu/Data/ShanghaiTech/part_A/train_data/images/IMG_19.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (800, 800))
    img = np.ascontiguousarray(img)
    transform = make_detr_transforms('val')

    output = inference_detector(model, transform(img).cuda(), None)
    result = postprocessors['bbox'](output, torch.tensor([[300, 300]]).cuda())

    print(result)

    # show the results
    img = cv2.imread('/home/hemingdu/Data/ShanghaiTech/part_A/train_data/images/IMG_19.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (300, 300))
    img = np.ascontiguousarray(img)

    for bbox in result['boxes']:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    plt.imshow(img)
    plt.show()