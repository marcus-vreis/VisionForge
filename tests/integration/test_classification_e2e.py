"""Opt-in end-to-end smoke test for the classification pipeline.

Runs the *real* ClassificationBlock (no mocks): builds a resnet18 (random init,
no downloads), trains one epoch on a tiny synthetic ImageFolder dataset, and
checks the full pipeline — ModelFactory → DataModule → Trainer → Evaluator →
MetricsPlotter → run.json. Skipped by default to keep CI fast (ADR-010); enable
with VF_RUN_CLASSIFICATION_INTEGRATION=1.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from visionforge.blocks.classification import ClassificationBlock
from visionforge.utils.config import ExperimentConfig

pytestmark = pytest.mark.skipif(
    not os.environ.get("VF_RUN_CLASSIFICATION_INTEGRATION"),
    reason="set VF_RUN_CLASSIFICATION_INTEGRATION=1 to run the classification e2e test",
)


def _synthetic_imagefolder(base: Path) -> None:
    rng = np.random.default_rng(0)
    for split in ("train", "val", "test"):
        for ci, cls in enumerate(("class_a", "class_b")):
            cdir = base / split / cls
            cdir.mkdir(parents=True, exist_ok=True)
            for i in range(3):
                # Class-correlated tint so the model has a signal to learn.
                arr = rng.integers(0, 80, (32, 32, 3), dtype=np.uint8)
                arr[..., ci] = rng.integers(160, 255, (32, 32), dtype=np.uint8)
                Image.fromarray(arr, "RGB").save(cdir / f"img{i}.png")


def test_classification_trains_end_to_end(tmp_path: Path) -> None:
    base = tmp_path / "ds"
    _synthetic_imagefolder(base)
    config = ExperimentConfig.model_validate(
        {
            "name": "e2e",
            "task": "multiclass",
            "model": {"name": "resnet18", "num_classes": 2, "pretrained": False},
            "training": {
                "epochs": 1,
                "batch_size": 2,
                "learning_rate": 0.001,
                "early_stopping_patience": 1,
                "seed": 0,
            },
            "data": {
                "base_dir": str(base),
                "num_workers": 0,
                "pin_memory": False,
                "transforms": {"image_size": 32},
            },
            "output": {"models_dir": str(tmp_path / "models")},
            "device": {"kind": "cpu"},
        }
    )

    block = ClassificationBlock()
    block.setup(config)
    block.run()
    report = block.report()

    assert "train" in report
    assert report["train"]["total_epochs"] == 1
    assert "eval" in report
    assert isinstance(report["eval"]["accuracy"], float)

    run_json_path = next((tmp_path / "models").rglob("run.json"))
    data = json.loads(run_json_path.read_text(encoding="utf-8"))
    assert data["status"] == "completed"
    assert "test_accuracy" in data["metrics"]
    assert isinstance(data["metrics"]["test_accuracy"], float)

    run_dir = run_json_path.parent
    assert (run_dir / "loss.png").is_file()
    assert (run_dir / "confusion_matrix.png").is_file()
    assert Path(data["artifacts"]["model"]).is_file()
