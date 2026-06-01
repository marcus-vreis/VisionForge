from __future__ import annotations

import pytest
import torch

from visionforge.core.detection_metrics import mean_average_precision_50


def _pred(boxes: list[list[float]], labels: list[int], scores: list[float]) -> dict:
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor(labels, dtype=torch.int64),
        "scores": torch.tensor(scores, dtype=torch.float32),
    }


def _gt(boxes: list[list[float]], labels: list[int]) -> dict:
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }


class TestMeanAveragePrecision50:
    def test_perfect_match_is_one(self) -> None:
        preds = [_pred([[0, 0, 10, 10]], [1], [0.9])]
        targets = [_gt([[0, 0, 10, 10]], [1])]
        result = mean_average_precision_50(preds, targets)
        assert result.map50 == pytest.approx(1.0)
        assert result.per_class[1] == pytest.approx(1.0)

    def test_no_predictions_is_zero(self) -> None:
        preds = [_pred([], [], [])]
        targets = [_gt([[0, 0, 10, 10]], [1])]
        assert mean_average_precision_50(preds, targets).map50 == 0.0

    def test_low_iou_is_false_positive(self) -> None:
        # Prediction nowhere near the ground truth → no true positive.
        preds = [_pred([[50, 50, 60, 60]], [1], [0.9])]
        targets = [_gt([[0, 0, 10, 10]], [1])]
        assert mean_average_precision_50(preds, targets).map50 == 0.0

    def test_empty_targets_is_zero(self) -> None:
        preds = [_pred([[0, 0, 10, 10]], [1], [0.9])]
        targets = [_gt([], [])]
        assert mean_average_precision_50(preds, targets).map50 == 0.0

    def test_half_correct_across_images(self) -> None:
        # img0: correct; img1: a false positive for the same class. Two GTs total
        # → recall caps at 0.5 with precision 1 → AP 0.5.
        preds = [
            _pred([[0, 0, 10, 10]], [1], [0.9]),
            _pred([[80, 80, 90, 90]], [1], [0.8]),
        ]
        targets = [
            _gt([[0, 0, 10, 10]], [1]),
            _gt([[0, 0, 10, 10]], [1]),
        ]
        assert mean_average_precision_50(preds, targets).map50 == pytest.approx(0.5)

    def test_mean_over_two_classes(self) -> None:
        # class 1 perfect (AP 1), class 2 missed (AP 0) → mAP 0.5.
        preds = [_pred([[0, 0, 10, 10]], [1], [0.9])]
        targets = [_gt([[0, 0, 10, 10], [20, 20, 30, 30]], [1, 2])]
        result = mean_average_precision_50(preds, targets)
        assert result.per_class[1] == pytest.approx(1.0)
        assert result.per_class[2] == pytest.approx(0.0)
        assert result.map50 == pytest.approx(0.5)

    def test_duplicate_detection_does_not_exceed_one(self) -> None:
        preds = [_pred([[0, 0, 10, 10], [0, 0, 10, 10]], [1, 1], [0.9, 0.8])]
        targets = [_gt([[0, 0, 10, 10]], [1])]
        assert mean_average_precision_50(preds, targets).map50 == pytest.approx(1.0)
