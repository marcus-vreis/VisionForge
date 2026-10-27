"""Typed fold contexts retain real training, progress and failure semantics."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from torch import nn

from visionforge.blocks import _cv_common as cv
from visionforge.core.cancellation import CancellationToken
from visionforge.core.data import _build_transforms
from visionforge.core.regression_data import RegressionCsvDataset
from visionforge.core.regression_trainer import RegressionTrainer
from visionforge.core.resume import RESUME_FILENAME
from visionforge.utils.regression_config import RegressionConfig


def _context(
    tmp_path: Path, *, shuffle: bool = False, token: CancellationToken | None = None
) -> cv.FoldContext:
    images = tmp_path / "data" / "images"
    images.mkdir(parents=True)
    for i in range(6):
        Image.new("RGB", (8, 8), (i * 30, 20, 30)).save(images / f"{i}.png")
    manifest = images.parent / "train.csv"
    manifest.write_text(
        "image,target\n" + "".join(f"{i}.png,{i / 6}\n" for i in range(6)),
        encoding="utf-8",
    )
    config = RegressionConfig.model_validate(
        {
            "name": "context_cv",
            "model": {"name": "resnet18", "num_targets": 1, "pretrained": False},
            "data": {
                "base_dir": str(images.parent),
                "target_columns": ["target"],
                "transforms": {"image_size": 32},
            },
            "training": {"epochs": 2, "batch_size": 2, "seed": 19},
            "output": {"models_dir": str(tmp_path / "models")},
            "device": {"kind": "cpu"},
        }
    )
    train_ds, val_ds = (
        RegressionCsvDataset(
            manifest,
            images,
            "image",
            ["target"],
            _build_transforms(config.data.transforms, is_train=is_train),
        )
        for is_train in (True, False)
    )
    return cv.FoldContext(
        config, train_ds, val_ds, cv.FoldSplit(3, shuffle, 23), None, token
    )


def _model(config: Any) -> nn.Module:
    return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 1))


def _metrics() -> cv.FoldMetrics:
    return cv.FoldMetrics(
        "r2",
        ("mse", "rmse", "mae", "r2"),
        ("r2", "rmse", "mae", "mse"),
        "Fold {}/{}: r2={:.4f} rmse={:.4f}",
        ("r2", "rmse"),
    )


@pytest.mark.parametrize("shuffle", [False, True])
def test_fold_context_preserves_metrics_artifacts_and_progress(
    tmp_path: Path, shuffle: bool
) -> None:
    """Each real fold publishes ordered metrics and its own complete run."""
    events: list[dict[str, Any]] = []
    context = replace(
        _context(tmp_path, shuffle=shuffle), progress_callback=events.append
    )
    report = cv.run_folds(context, cv.FoldTask(_model, RegressionTrainer, _metrics()))

    assert [fold.status for fold in report.folds] == ["success"] * 3
    assert report.metric == "r2"
    assert list(report.aggregate) == ["r2", "rmse", "mae", "mse"]
    assert sum(fold.val_size for fold in report.folds) == 6
    for fold in report.folds:
        assert (fold.train_size, fold.val_size) == (4, 2)
        assert list(fold.metrics) == ["mse", "rmse", "mae", "r2"]
        assert fold.run_dir is not None
        run_dir = Path(fold.run_dir)
        payload = json.loads((run_dir / "run.json").read_text("utf-8"))
        assert payload["experiment"] == f"context_cv_fold{fold.fold}"
        assert payload["metrics"]["total_epochs"] == 2
        assert (run_dir / "best_model.pth").is_file()
        assert not (run_dir / RESUME_FILENAME).exists()
    for aggregate in report.aggregate.values():
        assert aggregate["n"] == 3
        assert aggregate["std"] is not None
        assert aggregate["std_ddof"] == 1
    assert [
        event["trial_index"] for event in events if event["event"] == "trial_start"
    ] == [0, 1, 2]
    assert [
        event["trial_index"] for event in events if event["event"] == "trial_end"
    ] == [0, 1, 2]
    assert len([event for event in events if event["event"] == "epoch_end"]) == 6


def test_fold_context_isolates_factory_failure_and_closes_its_progress(
    tmp_path: Path,
) -> None:
    """Failure before training leaves a paired event and permits the next fold."""
    events: list[dict[str, Any]] = []
    context = replace(_context(tmp_path), progress_callback=events.append)
    calls = 0

    def fail_first_model(config: Any) -> nn.Module:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("model factory rejected fold")
        return _model(config)

    report = cv.run_folds(
        context, cv.FoldTask(fail_first_model, RegressionTrainer, _metrics())
    )
    assert [fold.status for fold in report.folds] == ["failed", "success", "success"]
    assert report.folds[0].error == "model factory rejected fold"
    assert report.folds[0].run_dir is None
    assert all(aggregate["n"] == 2 for aggregate in report.aggregate.values())
    assert events[0]["event"] == "trial_start"
    assert events[1] == {
        "event": "trial_end",
        "trial_index": 0,
        "total_trials": 3,
        "status": "failed",
    }


@pytest.mark.parametrize("stop_epoch", [0, 1])
def test_fold_context_discards_stopped_resume_and_excludes_aggregate(
    tmp_path: Path, stop_epoch: int
) -> None:
    """Stopped folds keep measured metrics but cannot resume outside their job."""
    token = CancellationToken()
    if stop_epoch == 0:
        token.cancel()

    def stop(event: dict[str, Any]) -> None:
        if event.get("event") == "epoch_end" and event["epoch"] == stop_epoch:
            token.cancel()

    context = replace(_context(tmp_path, token=token), progress_callback=stop)
    report = cv.run_folds(context, cv.FoldTask(_model, RegressionTrainer, _metrics()))
    assert len(report.folds) == 1
    fold = report.folds[0]
    assert fold.status == "stopped"
    assert bool(fold.metrics) is (stop_epoch == 1)
    assert report.aggregate == {}
    assert fold.run_dir is not None
    assert not (Path(fold.run_dir) / RESUME_FILENAME).exists()
    assert (Path(fold.run_dir) / "best_model.pth").exists() is (stop_epoch == 1)


def test_evaluation_failure_takes_precedence_over_stopped_fold(tmp_path: Path) -> None:
    """Scoring errors remain failures even when a real trainer reports a stop."""
    token = CancellationToken()

    class RejectEvaluation(RegressionTrainer):
        """Exercise the evaluation failure boundary after real fitting."""

        def evaluate(
            self, model: nn.Module, loader: Any
        ) -> tuple[float, float, float, float]:
            """Reject scoring after the fold has written its checkpoint."""
            raise ValueError("scoring rejected fold")

    def stop(event: dict[str, Any]) -> None:
        if event.get("event") == "epoch_end":
            token.cancel()

    context = replace(_context(tmp_path, token=token), progress_callback=stop)
    report = cv.run_folds(context, cv.FoldTask(_model, RejectEvaluation, _metrics()))
    assert len(report.folds) == 1
    fold = report.folds[0]
    assert fold.status == "failed"
    assert fold.error == "scoring rejected fold"
    assert fold.metrics == {}
    assert report.aggregate == {}
    assert fold.run_dir is not None
    assert (Path(fold.run_dir) / RESUME_FILENAME).is_file()
