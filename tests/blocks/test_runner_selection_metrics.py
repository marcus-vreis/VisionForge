"""Each built-in runner hands the sweep validation numbers, not test ones (ADR-131)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from visionforge.blocks import (
    classification_runner,
    detection_runner,
    regression_runner,
    segmentation_runner,
)
from visionforge.core.regression_trainer import (
    RegressionEpochResult,
    RegressionTrainResult,
)
from visionforge.core.segmentation_trainer import (
    SegmentationEpochResult,
    SegmentationTrainResult,
)
from visionforge.core.sweep import run_sweep
from visionforge.core.trainer import EpochResult, TrainResult


def _block_class(train_result: Any, report: dict[str, Any]) -> type:
    """A block that skips training and holds a finished trainer result."""

    class _Block:
        _cancel_token = None
        _progress_callback = None

        def setup(self, cfg: Any) -> None:
            self._train_result = train_result

        def run(self) -> None:
            pass

        def report(self) -> dict[str, Any]:
            return report

    return _Block


def _regression_epoch(epoch: int, r2: float) -> RegressionEpochResult:
    return RegressionEpochResult(
        epoch=epoch,
        train_loss=1.0,
        val_loss=1.0 - r2,
        val_mse=0.4 * epoch,
        val_rmse=0.2 * epoch,
        val_mae=0.1 * epoch,
        val_r2=r2,
    )


class TestRegressionRunner:
    def test_selects_on_the_kept_epochs_validation_metrics(
        self, tmp_path: Path
    ) -> None:
        train = RegressionTrainResult(
            best_epoch=2,
            best_val_loss=0.3,
            total_epochs=3,
            device_used="cpu",
            history=[
                _regression_epoch(e, r2) for e, r2 in ((1, 0.5), (2, 0.7), (3, 0.6))
            ],
        )
        report = {
            "train": {},
            "test": {"r2": 0.99, "rmse": 9.0, "mae": 8.0, "mse": 7.0},
        }
        with patch.object(
            regression_runner, "RegressionBlock", _block_class(train, report)
        ):
            result = regression_runner.RegressionRunner().run(object())

        assert result.metrics == {"r2": 0.99, "rmse": 9.0, "mae": 8.0, "mse": 7.0}
        assert result.selection_metrics == pytest.approx(
            {"r2": 0.7, "rmse": 0.4, "mae": 0.2, "mse": 0.8}
        )

    def test_no_epoch_means_no_validation_value(self) -> None:
        train = RegressionTrainResult(
            best_epoch=0, best_val_loss=float("inf"), total_epochs=0, device_used="cpu"
        )
        with patch.object(
            regression_runner, "RegressionBlock", _block_class(train, {"train": {}})
        ):
            result = regression_runner.RegressionRunner().run(object())

        assert result.selection_metrics == {}


class TestSegmentationRunner:
    def test_selects_on_the_kept_epochs_validation_metrics(self) -> None:
        history = [
            SegmentationEpochResult(e, 1.0, 1.0, miou, miou + 0.1, miou + 0.2)
            for e, miou in ((1, 0.3), (2, 0.6))
        ]
        train = SegmentationTrainResult(
            best_epoch=2,
            best_val_miou=0.6,
            total_epochs=2,
            device_used="cpu",
            history=history,
        )
        report = {"train": {}, "test": {"miou": 0.95, "dice": 0.96, "pixel_acc": 0.97}}
        with patch.object(
            segmentation_runner, "SegmentationBlock", _block_class(train, report)
        ):
            result = segmentation_runner.SegmentationRunner().run(object())

        assert result.metrics["miou"] == 0.95
        assert result.selection_metrics == pytest.approx(
            {"miou": 0.6, "dice": 0.7, "pixel_acc": 0.8}
        )


class TestClassificationRunner:
    def test_selects_on_validation_accuracy_only(self) -> None:
        train = TrainResult(
            best_epoch=1,
            best_val_loss=0.2,
            total_epochs=2,
            device_used="cpu",
            history=[
                EpochResult(1, 0.5, 0.8, 0.2, 0.75),
                EpochResult(2, 0.4, 0.9, 0.3, 0.8),
            ],
        )
        report = {"train": {}, "eval": {"accuracy": 0.99, "f1": 0.98, "auc_roc": 0.97}}
        with patch.object(
            classification_runner, "ClassificationBlock", _block_class(train, report)
        ):
            result = classification_runner.ClassificationRunner().run(object())

        assert result.metrics == {"accuracy": 0.99, "f1": 0.98, "auc_roc": 0.97}
        # The checkpoint kept is epoch 1's (lowest val loss), not the last one.
        assert result.selection_metrics == {"accuracy": 0.75}


class TestDetectionRunner:
    def test_its_metrics_are_already_validation_numbers(self) -> None:
        report = {"detection": {"best_map50_95": 0.4, "best_map50": 0.6}}
        with patch.object(
            detection_runner, "DetectionBlock", _block_class(None, report)
        ):
            result = detection_runner.DetectionRunner().run(object())

        assert (
            result.selection_metrics
            == result.metrics
            == {
                "map50_95": 0.4,
                "map50": 0.6,
            }
        )


class _Config:
    @classmethod
    def model_validate(cls, value: dict[str, Any]) -> dict[str, Any]:
        return value


def test_a_regression_sweep_ranks_on_validation_r2() -> None:
    """Test r2 favours the first trial, validation r2 the second: validation wins."""
    outcomes = {0.1: (0.5, 0.95), 0.2: (0.8, 0.10)}

    class _Runner(regression_runner.RegressionRunner):
        config_type = _Config  # type: ignore[assignment]  # a dict-backed stand-in

        def run(self, cfg: Any) -> Any:
            val_r2, test_r2 = outcomes[cfg["lr"]]
            train = RegressionTrainResult(
                best_epoch=1,
                best_val_loss=0.1,
                total_epochs=1,
                device_used="cpu",
                history=[_regression_epoch(1, val_r2)],
            )
            block = _block_class(train, {"train": {}, "test": {"r2": test_r2}})
            with patch.object(regression_runner, "RegressionBlock", block):
                return super().run(cfg)

    trials = run_sweep(
        _Runner(), {"lr": 0.1}, {"lr": [0.1, 0.2]}, mode="grid", metric="r2"
    )

    assert [t.overrides["lr"] for t in trials] == [0.2, 0.1]
    assert trials[0].metrics["r2"] == 0.8
    assert trials[0].evaluation_metrics["r2"] == 0.10
