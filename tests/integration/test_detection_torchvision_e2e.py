"""Opt-in end-to-end smoke test for the torchvision detection backend.

Runs the *real* DetectionTrainer (no mocks): builds a lightweight
fasterrcnn_mobilenet detector (random init, no downloads), trains one epoch on a
tiny synthetic YOLO dataset, and checks the full pipeline — loss loop, mAP@50
eval, checkpoint, run.json. Skipped by default to keep CI fast (ADR-010); enable
with VF_RUN_DETECTION_INTEGRATION=1.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from PIL import Image

from visionforge.core.detection_trainer import DetectionTrainer
from visionforge.utils.detection_config import DetectionConfig

pytestmark = pytest.mark.skipif(
    not os.environ.get("VF_RUN_DETECTION_INTEGRATION"),
    reason="set VF_RUN_DETECTION_INTEGRATION=1 to run the torchvision e2e smoke test",
)


def _synthetic_dataset(base: Path) -> None:
    for split in ("train", "val"):
        img_dir = base / "images" / split
        lbl_dir = base / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for i in range(2):
            Image.new("RGB", (128, 128), (60 + i * 30, 90, 120)).save(
                img_dir / f"img{i}.jpg"
            )
            (lbl_dir / f"img{i}.txt").write_text(
                "0 0.5 0.5 0.4 0.4\n", encoding="utf-8"
            )


def test_torchvision_backend_trains_end_to_end(tmp_path: Path) -> None:
    base = tmp_path / "ds"
    _synthetic_dataset(base)
    cfg = DetectionConfig.model_validate(
        {
            "name": "e2e",
            "model": {
                "backend": "torchvision",
                "name": "fasterrcnn_mobilenet_v3_large_fpn",
                "num_classes": 1,
                "pretrained": False,
            },
            "data": {"base_dir": str(base), "image_size": 128},
            "training": {
                "epochs": 1,
                "batch_size": 2,
                "learning_rate": 0.005,
                "workers": 0,
            },
            "output": {"models_dir": str(tmp_path / "models")},
            "device": {"kind": "cpu"},
        }
    )

    events: list[dict] = []
    result = DetectionTrainer(cfg).fit(progress_callback=events.append)

    assert result.model_path.exists()
    assert result.total_epochs == 1
    assert [e["event"] for e in events] == ["start", "epoch_end", "end"]

    run_json = json.loads((result.run_dir / "run.json").read_text("utf-8"))
    assert run_json["status"] == "completed"
    assert isinstance(run_json["metrics"]["map50"], float)
    assert run_json["metrics"]["box_loss"] is not None
