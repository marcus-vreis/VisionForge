from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch
from PIL import Image


def _write_image(path: Path, size: int = 64) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size), color=(120, 80, 40)).save(path)


def _make_regression_run(tmp_path: Path) -> Path:
    """Build a real regression run.json + random-init checkpoint on disk."""
    from visionforge.models.regression_factory import RegressionModelFactory
    from visionforge.utils.regression_config import RegressionModelConfig

    base_dir = tmp_path / "ds"
    base_dir.mkdir(parents=True, exist_ok=True)

    run_dir = tmp_path / "models" / "reg1" / "20260601_120000_000000"
    ckpt = run_dir / "weights" / "best.pth"
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    model = RegressionModelFactory.create(
        RegressionModelConfig(name="resnet18", num_targets=2, pretrained=False)
    )
    torch.save(model.state_dict(), ckpt)

    config = {
        "name": "reg1",
        "task": "regression",
        "model": {"name": "resnet18", "num_targets": 2, "pretrained": False},
        "data": {
            "base_dir": str(base_dir),
            "target_columns": ["height", "weight"],
            "transforms": {"image_size": 64},
        },
        "training": {"epochs": 1},
    }
    run_json = {
        "id": "reg1_20260601_120000_000000",
        "experiment": "reg1",
        "status": "completed",
        "run_dir": str(run_dir.resolve()),
        "config": config,
        "metrics": {"r2": 0.8},
        "history": [],
        "artifacts": {"model": str(ckpt.resolve()), "graphics": []},
        "tests": [],
    }
    (run_dir / "run.json").write_text(json.dumps(run_json), encoding="utf-8")
    return run_dir


class TestRegressionBatchPredict:
    def test_writes_target_columns_csv(self, tmp_path: Path) -> None:
        from visionforge.gui.api.routes import _execute_batch_predict
        from visionforge.gui.api.schemas import BatchPredictRequest

        run_dir = _make_regression_run(tmp_path)
        inbox = tmp_path / "inbox"
        _write_image(inbox / "a.png")
        _write_image(inbox / "b.png")

        resp = _execute_batch_predict(
            run_dir, BatchPredictRequest(input_dir=str(inbox), recursive=False)
        )

        assert resp.total_processed == 2
        assert resp.failed_count == 0

        out = Path(resp.output_csv)
        assert out.exists()
        with out.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == ["filename", "height", "weight"]
            rows = list(reader)
        assert len(rows) == 2
        # every target column holds a parseable float prediction
        for row in rows:
            float(row["height"])
            float(row["weight"])


class TestBatchPredictUnsupportedTasks:
    @staticmethod
    def _make_run(tmp_path: Path, task: str) -> Path:
        run_dir = tmp_path / "models" / task / "20260601_120000_000000"
        run_dir.mkdir(parents=True)
        run_json = {
            "config": {"name": task, "task": task, "model": {"name": "x"}},
            "artifacts": {"model": str(run_dir / "best.pt")},
        }
        (run_dir / "run.json").write_text(json.dumps(run_json), encoding="utf-8")
        return run_dir

    @pytest.mark.parametrize("task", ["segmentation", "detection"])
    def test_unsupported_task_is_rejected(self, tmp_path: Path, task: str) -> None:
        from visionforge.gui.api.routes import _execute_batch_predict
        from visionforge.gui.api.schemas import BatchPredictRequest

        run_dir = self._make_run(tmp_path, task)
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        with pytest.raises(ValueError, match=task):
            _execute_batch_predict(run_dir, BatchPredictRequest(input_dir=str(inbox)))
