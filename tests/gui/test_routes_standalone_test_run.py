""" "Test on another dataset" for the standalone tasks, against a real checkpoint.

The route only translates exceptions; the work is in ``_evaluate_standalone_run``,
which rebuilds the task's config around the folder the researcher chose. These
tests drive it with a randomly-initialised regression model — the metrics are
meaningless, but whether they are computed, recorded and validated is not.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch

from visionforge.gui.api.routes import _execute_run_test
from visionforge.gui.api.schemas import RunTestRequest
from visionforge.utils.selftest_data import (
    build_anomaly_dataset,
    build_regression_dataset,
    build_segmentation_dataset,
)


def _write_run(
    tmp_path: Path,
    experiment: str,
    config: dict[str, Any],
    model: torch.nn.Module | None,
) -> Path:
    """A run directory whose run.json points at a checkpoint (or at a missing one)."""
    run_dir = tmp_path / "models" / experiment / "20260923_120000_000000"
    ckpt = run_dir / "weights" / "best.pth"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    if model is not None:
        torch.save(model.state_dict(), ckpt)

    run_json = {
        "id": f"{experiment}_20260923_120000_000000",
        "experiment": experiment,
        "status": "completed",
        "run_dir": str(run_dir.resolve()),
        "config": config,
        "metrics": {},
        "history": [],
        "artifacts": {"model": str(ckpt.resolve())},
        "tests": [],
    }
    (run_dir / "run.json").write_text(json.dumps(run_json), encoding="utf-8")
    return run_dir


def _regression_run(tmp_path: Path, *, with_checkpoint: bool = True) -> Path:
    """A run.json + resnet18 checkpoint trained on nothing, over a synthetic set."""
    from visionforge.models.regression_factory import RegressionModelFactory
    from visionforge.utils.regression_config import RegressionModelConfig

    base = build_regression_dataset(tmp_path / "ds", size=32, rows=12)
    model = (
        RegressionModelFactory.create(
            RegressionModelConfig(name="resnet18", num_targets=1, pretrained=False)
        )
        if with_checkpoint
        else None
    )
    config: dict[str, Any] = {
        "name": "reg",
        "task": "regression",
        "model": {"name": "resnet18", "num_targets": 1, "pretrained": False},
        "data": {
            "base_dir": str(base),
            "target_columns": ["target"],
            "image_size": 32,
            "num_workers": 0,
            "pin_memory": False,
        },
        "training": {"epochs": 1, "batch_size": 4},
        "device": {"kind": "cpu"},
    }
    return _write_run(tmp_path, "reg", config, model)


def _segmentation_run(tmp_path: Path) -> Path:
    """A run.json + random-init U-Net checkpoint over a synthetic paired set."""
    from visionforge.models.segmentation_factory import SegmentationModelFactory
    from visionforge.utils.segmentation_config import SegmentationModelConfig

    base = build_segmentation_dataset(tmp_path / "ds", size=32, pairs=4)
    model_cfg = {"name": "unet", "num_classes": 3, "pretrained": False}
    model = SegmentationModelFactory.create(
        SegmentationModelConfig.model_validate(model_cfg)
    )
    config: dict[str, Any] = {
        "name": "seg",
        "task": "segmentation",
        "model": model_cfg,
        "data": {
            "base_dir": str(base),
            "image_size": 32,
            "num_workers": 0,
            "pin_memory": False,
        },
        "training": {"epochs": 1, "batch_size": 2},
        "device": {"kind": "cpu"},
    }
    return _write_run(tmp_path, "seg", config, model)


def _anomaly_run(tmp_path: Path) -> Path:
    """A run.json whose checkpoint is an empty file: only the data check is reached."""
    base = build_anomaly_dataset(tmp_path / "ds", size=32, normals=4)
    config: dict[str, Any] = {
        "name": "anom",
        "task": "anomaly",
        "model": {"name": "autoencoder", "latent_dim": 8},
        "data": {
            "base_dir": str(base),
            "image_size": 32,
            "num_workers": 0,
            "pin_memory": False,
        },
        "training": {"epochs": 1, "batch_size": 2},
        "device": {"kind": "cpu"},
    }
    run_dir = _write_run(tmp_path, "anom", config, None)
    (run_dir / "weights" / "best.pth").write_bytes(b"")
    return run_dir


class TestRegressionTestRun:
    def test_scores_the_chosen_manifest_and_records_it(self, tmp_path: Path) -> None:
        run_dir = _regression_run(tmp_path)
        # Another dataset, under another manifest name, with the run's own
        # dataset gone: only the redirect to the chosen file can evaluate this.
        other = build_regression_dataset(tmp_path / "other", size=32, rows=12)
        manifest = other / "holdout.csv"
        (other / "test.csv").rename(manifest)
        shutil.rmtree(tmp_path / "ds")

        resp = _execute_run_test(
            run_dir, RunTestRequest(data_dir=str(manifest), label="held-out")
        )

        assert set(resp.metrics) == {"mse", "rmse", "mae", "r2"}
        assert all(math.isfinite(v) for v in resp.metrics.values())
        assert resp.metrics["rmse"] == pytest.approx(math.sqrt(resp.metrics["mse"]))
        assert resp.label == "held-out"
        saved = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        assert len(saved["tests"]) == 1
        assert saved["tests"][0]["metrics"] == resp.metrics
        assert saved["tests"][0]["base_dir"] == str(manifest.resolve())

    def test_a_manifest_alone_in_its_folder_is_enough(self, tmp_path: Path) -> None:
        run_dir = _regression_run(tmp_path)
        # What a held-out set looks like in practice: one manifest and its
        # images, with no train.csv / val.csv beside it.
        holdout = tmp_path / "holdout"
        build_regression_dataset(tmp_path / "scratch", size=32, rows=12)
        shutil.copytree(tmp_path / "scratch" / "images", holdout / "images")
        shutil.copy(tmp_path / "scratch" / "test.csv", holdout / "holdout.csv")
        assert sorted(p.name for p in holdout.iterdir()) == ["holdout.csv", "images"]

        resp = _execute_run_test(
            run_dir, RunTestRequest(data_dir=str(holdout / "holdout.csv"))
        )

        assert set(resp.metrics) == {"mse", "rmse", "mae", "r2"}
        assert all(math.isfinite(v) for v in resp.metrics.values())

    def test_the_label_defaults_to_the_file_name(self, tmp_path: Path) -> None:
        run_dir = _regression_run(tmp_path)

        resp = _execute_run_test(
            run_dir, RunTestRequest(data_dir=str(tmp_path / "ds" / "test.csv"))
        )

        assert resp.label == "test.csv"

    def test_a_folder_is_refused_because_regression_reads_a_manifest(
        self, tmp_path: Path
    ) -> None:
        # No checkpoint on purpose: the folder complaint must come first.
        run_dir = _regression_run(tmp_path, with_checkpoint=False)

        with pytest.raises(ValueError, match="manifesto"):
            _execute_run_test(run_dir, RunTestRequest(data_dir=str(tmp_path / "ds")))

    def test_a_missing_path_is_named_before_the_checkpoint_is_checked(
        self, tmp_path: Path
    ) -> None:
        # Both are wrong here; the path the researcher just typed is reported,
        # because a complaint about the checkpoint would send them the wrong way.
        run_dir = _regression_run(tmp_path, with_checkpoint=False)

        with pytest.raises(ValueError, match="não encontrado"):
            _execute_run_test(
                run_dir, RunTestRequest(data_dir=str(tmp_path / "nope.csv"))
            )

    def test_a_run_without_its_checkpoint_is_a_missing_file(
        self, tmp_path: Path
    ) -> None:
        run_dir = _regression_run(tmp_path, with_checkpoint=False)

        with pytest.raises(FileNotFoundError, match="checkpoint"):
            _execute_run_test(
                run_dir, RunTestRequest(data_dir=str(tmp_path / "ds" / "test.csv"))
            )


class TestSegmentationTestRun:
    def test_a_split_folder_alone_is_enough(self, tmp_path: Path) -> None:
        run_dir = _segmentation_run(tmp_path)
        # The held-out folder carries its own images/ and masks/ and nothing
        # else: no train/ or val/ sibling to load.
        other = build_segmentation_dataset(tmp_path / "other", size=32, pairs=4)
        holdout = tmp_path / "holdout"
        shutil.copytree(other / "test", holdout)
        assert sorted(p.name for p in holdout.iterdir()) == ["images", "masks"]

        resp = _execute_run_test(run_dir, RunTestRequest(data_dir=str(holdout)))

        assert set(resp.metrics) == {"miou", "dice", "pixel_acc"}
        assert all(math.isfinite(v) for v in resp.metrics.values())
        assert resp.label == "holdout"

    def test_a_folder_without_pairs_is_named(self, tmp_path: Path) -> None:
        run_dir = _segmentation_run(tmp_path)
        empty = tmp_path / "empty"
        empty.mkdir()

        with pytest.raises(ValueError, match="Nenhum par imagem/máscara"):
            _execute_run_test(run_dir, RunTestRequest(data_dir=str(empty)))


class TestAnomalyTestRun:
    def test_a_missing_normal_train_split_explains_why_it_is_needed(
        self, tmp_path: Path
    ) -> None:
        run_dir = _anomaly_run(tmp_path)
        # A test folder on its own: the threshold has nothing to be calibrated on.
        other = build_anomaly_dataset(tmp_path / "other", size=32, normals=4)
        holdout = tmp_path / "holdout"
        shutil.copytree(other / "test", holdout)

        with pytest.raises(ValueError, match="limiar") as excinfo:
            _execute_run_test(run_dir, RunTestRequest(data_dir=str(holdout)))

        message = str(excinfo.value)
        assert str(tmp_path / "train" / "good") in message
        assert "ao lado" in message
