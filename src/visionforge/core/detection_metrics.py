"""mAP@0.5 for object detection (VOC all-points AP).

Pure, dependency-light metric used to select and report torchvision detectors
(ADR-035 follow-up). Predictions and targets follow the torchvision format:
each is a list (per image) of dicts with ``boxes`` (xyxy) and ``labels``;
predictions also carry ``scores``. Background (label 0) never appears in targets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torchvision.ops import box_iou

Detection = dict[str, torch.Tensor]


@dataclass
class MapResult:
    """mAP@0.5 plus the per-class average precision it averages."""

    map50: float
    per_class: dict[int, float]


def _ap_all_points(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """VOC all-points AP: area under the monotonic precision envelope."""
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _ap_for_class(
    predictions: list[Detection],
    targets: list[Detection],
    class_id: int,
    iou_threshold: float,
) -> float | None:
    """Average precision for one class, or None if the class has no ground truth."""
    n_gt = sum(int((t["labels"] == class_id).sum()) for t in targets)
    if n_gt == 0:
        return None

    entries: list[tuple[float, int, torch.Tensor]] = []
    for img_idx, pred in enumerate(predictions):
        mask = pred["labels"] == class_id
        for box, score in zip(pred["boxes"][mask], pred["scores"][mask], strict=True):
            entries.append((float(score), img_idx, box))
    if not entries:
        return 0.0

    entries.sort(key=lambda e: e[0], reverse=True)
    matched: dict[int, set[int]] = {i: set() for i in range(len(targets))}
    tp = np.zeros(len(entries))
    fp = np.zeros(len(entries))

    for i, (_score, img_idx, box) in enumerate(entries):
        gt_mask = targets[img_idx]["labels"] == class_id
        gt_boxes = targets[img_idx]["boxes"][gt_mask]
        if gt_boxes.numel() == 0:
            fp[i] = 1
            continue
        ious = box_iou(box.unsqueeze(0), gt_boxes)[0]
        best_iou = float(ious.max())
        best_j = int(ious.argmax())
        if best_iou >= iou_threshold and best_j not in matched[img_idx]:
            tp[i] = 1
            matched[img_idx].add(best_j)
        else:
            fp[i] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / n_gt
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    return _ap_all_points(recalls, precisions)


def mean_average_precision_50(
    predictions: list[Detection],
    targets: list[Detection],
    iou_threshold: float = 0.5,
) -> MapResult:
    """Compute mAP@0.5 over every class present in ``targets``.

    Classes with no ground truth are ignored. Returns map50 = 0.0 when there is
    no ground truth at all (e.g. an empty validation set).
    """
    class_ids: set[int] = set()
    for t in targets:
        class_ids.update(int(x) for x in t["labels"].tolist())

    per_class: dict[int, float] = {}
    for c in sorted(class_ids):
        ap = _ap_for_class(predictions, targets, c, iou_threshold)
        if ap is not None:
            per_class[c] = ap

    map50 = float(np.mean(list(per_class.values()))) if per_class else 0.0
    return MapResult(map50=map50, per_class=per_class)


__all__ = ["MapResult", "mean_average_precision_50"]
