from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from visionforge.blocks import detection as det_mod
from visionforge.blocks.detection import DetectionBlock
from visionforge.core.cancellation import CancellationToken
from visionforge.core.detection_trainer import DetectionTrainResult
from visionforge.utils.detection_config import DetectionConfig


def _config(tmp_path: Path) -> DetectionConfig:
    base = tmp_path / "ds"
    for split in ("train", "val"):
        (base / "images" / split).mkdir(parents=True, exist_ok=True)
    return DetectionConfig.model_validate(
        {
            "name": "det",
            "model": {"name": "yolo11n", "num_classes": 2},
            "data": {"base_dir": str(base)},
            "training": {"epochs": 1, "batch_size": 8, "learning_rate": 0.01},
            "output": {"models_dir": str(tmp_path / "models")},
        }
    )


def _fake_trainer(captured: dict[str, Any], tmp_path: Path) -> type:
    class _FakeTrainer:
        def __init__(self, config: DetectionConfig) -> None:
            captured["config"] = config

        def fit(
            self, progress_callback: Any = None, cancel_token: Any = None
        ) -> DetectionTrainResult:
            captured["callback"] = progress_callback
            captured["cancel_token"] = cancel_token
            return DetectionTrainResult(
                best_epoch=2,
                best_map50_95=0.4,
                total_epochs=2,
                device_used="cpu",
                history=[],
                model_path=tmp_path / "weights" / "best.pt",
                run_dir=tmp_path / "run",
            )

    return _FakeTrainer


class TestDetectionBlock:
    def test_report_before_run_is_empty(self, tmp_path: Path) -> None:
        block = DetectionBlock()
        block.setup(_config(tmp_path))
        assert block.report() == {}

    def test_run_invokes_trainer_and_reports(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}
        monkeypatch.setattr(
            det_mod, "DetectionTrainer", _fake_trainer(captured, tmp_path)
        )
        block = DetectionBlock()
        block.setup(_config(tmp_path))
        block.run()

        rep = block.report()
        assert rep["detection"]["best_map50_95"] == 0.4
        assert rep["detection"]["best_epoch"] == 2
        assert rep["detection"]["model_path"].endswith("best.pt")

    def test_progress_callback_forwarded_to_trainer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}
        monkeypatch.setattr(
            det_mod, "DetectionTrainer", _fake_trainer(captured, tmp_path)
        )
        block = DetectionBlock()
        block.setup(_config(tmp_path))

        def cb(_event: dict[str, Any]) -> None:
            pass

        block._progress_callback = cb
        block.run()
        assert captured["callback"] is cb


class TestCancellationReachesTheTrainer:
    """The queue's stop request is worthless unless the loop receives it."""

    def test_the_blocks_token_is_handed_to_the_trainer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}
        monkeypatch.setattr(
            det_mod, "DetectionTrainer", _fake_trainer(captured, tmp_path)
        )
        token = CancellationToken()
        block = DetectionBlock()
        block.setup(_config(tmp_path))
        block._cancel_token = token
        block.run()

        assert captured["cancel_token"] is token
