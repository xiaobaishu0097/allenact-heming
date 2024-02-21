import os

import h5py
import torch
import util.misc as utils
from datasets.coco_eval import CocoEvaluator


@torch.no_grad()
def extract_feature(
    model, criterion, postprocessors, data_loader, base_ds, device, output_dir
):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
    )
    header = "Test:"

    iou_types = tuple(k for k in ("segm", "bbox") if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples, feature_extraction=True)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        metric_logger.update(
            loss=sum(loss_dict_reduced_scaled.values()),
            **loss_dict_reduced_scaled,
            **loss_dict_reduced_unscaled,
        )
        metric_logger.update(class_error=loss_dict_reduced["class_error"])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors["bbox"](outputs, orig_target_sizes)
        # res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        for index, target in enumerate(targets):
            features_file_path = os.path.join(
                output_dir, "{:09d}.hdf5".format(target["image_id"].item())
            )
            with h5py.File(features_file_path, "w") as wf:
                wf.create_dataset(
                    "features", data=outputs["encoder_features"][index].cpu().numpy()
                )
                predicted = wf.create_group("predicted")
                predicted.create_dataset(
                    "scores",
                    data=outputs["pred_logits"][index].max(-1)[0].cpu().numpy(),
                )
                predicted.create_dataset(
                    "labels",
                    data=outputs["pred_logits"][index].max(-1)[1].cpu().numpy(),
                )
                estimated = wf.create_group("estimated")
                estimated.create_dataset(
                    "scores", data=results[index]["scores"].cpu().numpy()
                )
                estimated.create_dataset(
                    "labels", data=results[index]["labels"].cpu().numpy()
                )
                wf.create_dataset("bboxes", data=results[index]["boxes"].cpu().numpy())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    return
