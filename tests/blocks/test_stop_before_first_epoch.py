"""A run stopped before its first epoch ends cleanly in every block (ADR-111).

The trainers already broke out of their loop before epoch 1; the blocks then
reloaded a checkpoint that was never written, and the run died with a
FileNotFoundError. Each family is driven for real on a tiny synthetic dataset
with the stop already pressed, which is the stop that lands while a run is
still loading its data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from visionforge.core.cancellation import CancellationToken
from visionforge.core.resume import RESUME_FILENAME
from visionforge.utils.selftest_data import (
    build_anomaly_dataset,
    build_classification_dataset,
    build_regression_dataset,
    build_segmentation_dataset,
)

EPOCHS = 3


def _stopped() -> CancellationToken:
    token = CancellationToken()
    token.cancel()
    return token


def _strict_json(path: Path) -> dict[str, Any]:
    """Parse run.json refusing the NaN/Infinity tokens no browser can read."""

    def _reject(constant: str) -> None:
        raise AssertionError(f"{path.name} carries {constant}, which is not JSON")

    data: dict[str, Any] = json.loads(
        path.read_text(encoding="utf-8"), parse_constant=_reject
    )
    return data


def _output(tmp_path: Path) -> dict[str, str]:
    return {
        "models_dir": str(tmp_path / "models"),
        "reports_dir": str(tmp_path / "reports"),
        "graphics_dir": str(tmp_path / "graphics"),
        "logs_dir": str(tmp_path / "logs"),
    }


def _assert_empty_run(
    run_dir: Path, events: list[dict[str, Any]], report: dict[str, Any]
) -> dict[str, Any]:
    """What every family must leave behind: an honest record of nothing done."""
    from visionforge.gui.api.routes import _resume_status

    # PatchCore announces its first phase before it reads the token; a phase
    # is progress, not training, so only the run's own frame is checked.
    assert [e["event"] for e in events if e["event"] != "phase"] == ["start", "end"]
    assert events[-1]["total_epochs"] == 0
    json.dumps(report, allow_nan=False)  # the result endpoint serves this

    data = _strict_json(run_dir / "run.json")
    assert data["metrics"]["total_epochs"] == 0
    assert data["history"] == []
    assert not (run_dir / "best_model.pth").exists()
    assert data["artifacts"]["model"] is None
    # Nothing ran, so there is nothing to continue: no resume file, and the
    # history says so the same way it does for any other run.
    assert not (run_dir / RESUME_FILENAME).exists()
    assert _resume_status(run_dir, data) == (False, EPOCHS)
    return data


def _drive(block: Any, config: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    block.setup(config)
    block._progress_callback = events.append
    block._cancel_token = _stopped()
    block.run()
    return block.report(), events


class TestClassificationFamily:
    def _config(self, tmp_path: Path, **extra: Any) -> Any:
        from visionforge.utils.config import ExperimentConfig

        base = build_classification_dataset(tmp_path / "ds")
        return ExperimentConfig.model_validate(
            {
                "name": "stop_cls",
                "model": {"name": "resnet18", "num_classes": 2, "pretrained": False},
                "training": {"epochs": EPOCHS, "batch_size": 4, "seed": 0},
                "data": {
                    "base_dir": str(base),
                    "num_workers": 0,
                    "pin_memory": False,
                    "transforms": {"image_size": 32},
                },
                "output": _output(tmp_path),
                "device": {"kind": "cpu"},
                **extra,
            }
        )

    def test_classification(self, tmp_path: Path) -> None:
        from visionforge.blocks.classification import ClassificationBlock

        block = ClassificationBlock()
        report, events = _drive(block, self._config(tmp_path))

        assert report["train"]["total_epochs"] == 0
        assert report["train"]["best_val_loss"] is None
        assert "eval" not in report
        assert block._train_result is not None
        data = _assert_empty_run(block._train_result.model_path.parent, events, report)
        assert data["metrics"]["best_val_loss"] is None

    def test_transfer_learning(self, tmp_path: Path) -> None:
        from visionforge.blocks.transfer_learning import TransferLearningBlock

        config = self._config(
            tmp_path,
            block="transfer_learning",
            transfer_learning={"mode": "feature_extraction"},
        )
        block = TransferLearningBlock()
        report, events = _drive(block, config)

        assert report["train"]["total_epochs"] == 0
        assert "eval" not in report
        assert block._train_result is not None
        _assert_empty_run(block._train_result.model_path.parent, events, report)


class TestStandaloneFamilies:
    def test_regression(self, tmp_path: Path) -> None:
        from visionforge.blocks.regression import RegressionBlock
        from visionforge.utils.regression_config import RegressionConfig

        base = build_regression_dataset(tmp_path / "ds")
        config = RegressionConfig.model_validate(
            {
                "name": "stop_reg",
                "model": {"name": "resnet18", "num_targets": 1, "pretrained": False},
                "data": {
                    "base_dir": str(base),
                    "target_columns": ["target"],
                    "image_size": 32,
                    "num_workers": 0,
                    "pin_memory": False,
                },
                "training": {"epochs": EPOCHS, "batch_size": 4, "seed": 0},
                "output": _output(tmp_path),
                "device": {"kind": "cpu"},
            }
        )
        report, events = _drive(RegressionBlock(), config)

        assert report["train"]["best_val_loss"] is None
        assert "test" not in report
        data = _assert_empty_run(Path(report["train"]["run_dir"]), events, report)
        assert data["metrics"]["r2"] is None  # not a 0.0 that reads as measured

    def test_segmentation(self, tmp_path: Path) -> None:
        from visionforge.blocks.segmentation import SegmentationBlock
        from visionforge.utils.segmentation_config import SegmentationConfig

        base = build_segmentation_dataset(tmp_path / "ds")
        config = SegmentationConfig.model_validate(
            {
                "name": "stop_seg",
                "model": {"name": "unet", "num_classes": 3, "pretrained": False},
                "data": {
                    "base_dir": str(base),
                    "image_size": 64,
                    "num_workers": 0,
                    "pin_memory": False,
                },
                "training": {"epochs": EPOCHS, "batch_size": 2, "seed": 0},
                "output": _output(tmp_path),
                "device": {"kind": "cpu"},
            }
        )
        report, events = _drive(SegmentationBlock(), config)

        # The degenerate-run guard used to save the untrained weights here.
        assert report["train"]["best_val_miou"] is None
        assert "test" not in report
        data = _assert_empty_run(Path(report["train"]["run_dir"]), events, report)
        assert data["metrics"]["miou"] is None

    @pytest.mark.parametrize("model_name", ["autoencoder", "patchcore"])
    def test_anomaly(self, tmp_path: Path, model_name: str) -> None:
        from visionforge.blocks.anomaly import AnomalyBlock
        from visionforge.utils.anomaly_config import AnomalyConfig

        base = build_anomaly_dataset(tmp_path / "ds")
        model: dict[str, Any] = (
            {"name": "autoencoder", "latent_dim": 16}
            if model_name == "autoencoder"
            else {
                "name": "patchcore",
                "backbone": "resnet18",
                "pretrained": False,
                "coreset_ratio": 0.5,
            }
        )
        config = AnomalyConfig.model_validate(
            {
                "name": f"stop_{model_name}",
                "model": model,
                "data": {
                    "base_dir": str(base),
                    "image_size": 64,
                    "num_workers": 0,
                    "pin_memory": False,
                },
                "training": {"epochs": EPOCHS, "batch_size": 2, "seed": 0},
                "output": _output(tmp_path),
                "device": {"kind": "cpu"},
            }
        )
        report, events = _drive(AnomalyBlock(), config)

        assert report["train"]["best_auroc"] is None
        assert "test" not in report
        data = _assert_empty_run(Path(report["train"]["run_dir"]), events, report)
        assert data["metrics"]["auroc"] is None
