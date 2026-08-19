"""The history has to show each task the numbers that task actually writes.

Found by running the five tasks and reading the list: segmentation and anomaly
showed an empty metric cell and regression showed its loss, because the summary
fell back to the classification keys — which none of them write. Every one of
these headline numbers was on disk in run.json the whole time.
"""

from __future__ import annotations

from typing import Any

from visionforge.gui.api.routes import _summary_metrics


class TestEachTaskSurfacesItsOwnMetrics:
    def test_segmentation_shows_miou_dice_and_pixel_accuracy(self) -> None:
        metrics = {
            "best_val_miou": 0.63,
            "test_miou": 0.6118,
            "test_dice": 0.7573,
            "test_pixel_acc": 0.7670,
        }

        assert _summary_metrics("segmentation", metrics) == {
            "miou": 0.6118,
            "dice": 0.7573,
            "pixel_acc": 0.7670,
        }

    def test_regression_shows_r2_not_the_raw_loss(self) -> None:
        metrics = {
            "best_val_loss": 101.77,
            "test_r2": 0.5185,
            "test_mae": 8.8966,
            "test_rmse": 11.1605,
        }

        got = _summary_metrics("regression", metrics)

        assert got == {"r2": 0.5185, "mae": 8.8966, "rmse": 11.1605}
        assert "val_loss" not in got

    def test_anomaly_shows_auroc_and_image_f1(self) -> None:
        metrics = {
            "best_auroc": 0.79,
            "test_auroc": 0.7908,
            "test_image_f1": 0.0,
            "test_threshold": 0.032,
        }

        # An F1 of 0.0 is a real reading, not a missing one: it has to survive.
        assert _summary_metrics("anomaly", metrics) == {
            "auroc": 0.7908,
            "f1": 0.0,
        }

    def test_classification_and_detection_are_unchanged(self) -> None:
        assert _summary_metrics(
            "classification",
            {"test_accuracy": 0.82, "test_f1": 0.81, "best_val_loss": 0.4},
        ) == {"accuracy": 0.82, "f1": 0.81, "val_loss": 0.4}
        assert _summary_metrics("detection", {"map50": 0.65, "map50_95": 0.35}) == {
            "map50": 0.65,
            "map50_95": 0.35,
        }


class TestBlockLabel:
    """A standalone task's block is its task, not the classification default."""

    @staticmethod
    def _summary(task: str, config: dict[str, Any]) -> str:
        from datetime import datetime
        from pathlib import Path

        from visionforge.gui.api.routes import _parse_run_summary

        data = {
            "experiment": "e",
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "status": "completed",
            "config": config,
            "metrics": {"total_epochs": 1},
            "history": [],
            "artifacts": {},
        }
        return _parse_run_summary(Path("runs/x"), data).block

    def test_segmentation_is_not_filed_as_classification(self) -> None:
        assert self._summary("segmentation", {"task": "segmentation"}) == "segmentation"

    def test_anomaly_and_regression_too(self) -> None:
        assert self._summary("anomaly", {"task": "anomaly"}) == "anomaly"
        assert self._summary("regression", {"task": "regression"}) == "regression"

    def test_an_explicit_block_still_wins(self) -> None:
        """Classification's sweeps and folds declare their own block."""
        assert (
            self._summary("multiclass", {"task": "multiclass", "block": "grid_search"})
            == "grid_search"
        )
