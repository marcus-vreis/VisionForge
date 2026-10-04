""" "Test on another dataset" for the standalone tasks, against a real checkpoint.

The route only translates exceptions; the work is in ``_evaluate_standalone_run``,
which rebuilds the task's config around the folder the researcher chose. These
tests drive it with a randomly-initialised regression model — the metrics are
meaningless, but whether they are computed, recorded and validated is not.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from visionforge.gui.api.routes import _execute_run_test
from visionforge.gui.api.schemas import RunTestRequest
from visionforge.utils.selftest_data import build_regression_dataset


def _regression_run(tmp_path: Path, *, with_checkpoint: bool = True) -> Path:
    """A run.json + resnet18 checkpoint trained on nothing, over a synthetic set."""
    from visionforge.models.regression_factory import RegressionModelFactory
    from visionforge.utils.regression_config import RegressionModelConfig

    base = build_regression_dataset(tmp_path / "ds", size=32, rows=12)

    run_dir = tmp_path / "models" / "reg" / "20260923_120000_000000"
    ckpt = run_dir / "weights" / "best.pth"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    if with_checkpoint:
        model = RegressionModelFactory.create(
            RegressionModelConfig(name="resnet18", num_targets=1, pretrained=False)
        )
        torch.save(model.state_dict(), ckpt)

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
    run_json = {
        "id": "reg_20260923_120000_000000",
        "experiment": "reg",
        "status": "completed",
        "run_dir": str(run_dir.resolve()),
        "config": config,
        "metrics": {"r2": 0.1},
        "history": [],
        "artifacts": {"model": str(ckpt.resolve())},
        "tests": [],
    }
    (run_dir / "run.json").write_text(json.dumps(run_json), encoding="utf-8")
    return run_dir


class TestRegressionTestRun:
    def test_scores_the_chosen_manifest_and_records_it(self, tmp_path: Path) -> None:
        run_dir = _regression_run(tmp_path)
        manifest = tmp_path / "ds" / "test.csv"

        resp = _execute_run_test(
            run_dir, RunTestRequest(data_dir=str(manifest), label="held-out")
        )

        assert set(resp.metrics) == {"mse", "rmse", "mae", "r2"}
        assert resp.label == "held-out"
        saved = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        assert len(saved["tests"]) == 1
        assert saved["tests"][0]["metrics"] == resp.metrics
        assert saved["tests"][0]["base_dir"] == str(manifest.resolve())

    def test_the_label_defaults_to_the_file_name(self, tmp_path: Path) -> None:
        run_dir = _regression_run(tmp_path)

        resp = _execute_run_test(
            run_dir, RunTestRequest(data_dir=str(tmp_path / "ds" / "test.csv"))
        )

        assert resp.label == "test.csv"

    def test_a_folder_is_refused_because_regression_reads_a_manifest(
        self, tmp_path: Path
    ) -> None:
        run_dir = _regression_run(tmp_path)

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
