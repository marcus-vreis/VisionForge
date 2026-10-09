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


class TestRunWithoutATestSplit:
    """A run that never scored a held-out split still has a card.

    Every trainer writes the validation score at the best epoch under the bare
    name (``r2``, ``miou``, ``auroc``); only ``test_*`` appears once a test
    split was scored. Projecting ``test_*`` alone left such a run's card empty,
    though the numbers were on disk. The validation value is sent under
    ``val_<label>``, so the card can say which split it is.
    """

    def test_regression_falls_back_to_the_validation_scores(self) -> None:
        metrics = {"best_val_loss": 12.0, "r2": 0.41, "mae": 3.2, "rmse": 4.5}

        assert _summary_metrics("regression", metrics) == {
            "val_r2": 0.41,
            "val_mae": 3.2,
            "val_rmse": 4.5,
        }

    def test_segmentation_falls_back_to_the_validation_scores(self) -> None:
        metrics = {"miou": 0.5, "dice": 0.6, "pixel_acc": 0.7}

        assert _summary_metrics("segmentation", metrics) == {
            "val_miou": 0.5,
            "val_dice": 0.6,
            "val_pixel_acc": 0.7,
        }

    def test_anomaly_falls_back_through_its_image_f1(self) -> None:
        metrics = {"auroc": 0.88, "image_f1": 0.0, "threshold": 0.3}

        # The projected name is the headline's (`f1`), the source is `image_f1`.
        assert _summary_metrics("anomaly", metrics) == {
            "val_auroc": 0.88,
            "val_f1": 0.0,
        }

    def test_a_test_value_wins_over_the_validation_one(self) -> None:
        metrics = {"r2": 0.41, "test_r2": 0.52, "mae": 3.2}

        # Per metric: r2 was scored on the test split, mae was not.
        assert _summary_metrics("regression", metrics) == {"r2": 0.52, "val_mae": 3.2}

    def test_a_metric_never_measured_is_still_left_out(self) -> None:
        metrics = {"r2": None, "mae": "n/a", "rmse": 4.5}

        assert _summary_metrics("regression", metrics) == {"val_rmse": 4.5}

    def test_detection_has_no_test_split_so_keeps_its_names(self) -> None:
        assert _summary_metrics("detection", {"map50": 0.6}) == {"map50": 0.6}

    def test_a_classification_run_keeps_its_validation_loss(self) -> None:
        """Classification writes no bare validation accuracy: only its loss."""
        assert _summary_metrics("classification", {"best_val_loss": 0.4}) == {
            "val_loss": 0.4
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
