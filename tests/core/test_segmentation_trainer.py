from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from visionforge.core.segmentation_trainer import (
    SegmentationMetricAccumulator,
    SegmentationTrainer,
    dice_loss,
)
from visionforge.utils.segmentation_config import SegmentationConfig


class TinySegModel(nn.Module):
    """1x1 conv: (B, 3, H, W) -> (B, num_classes, H, W). Cheap for trainer tests."""

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class FakeSegDataModule:
    """Fixed batches of (image, mask) for trainer tests."""

    def __init__(
        self, num_classes: int = 3, size: int = 16, n_batches: int = 2
    ) -> None:
        self._c = num_classes
        self._s = size
        self._n = n_batches

    def _batches(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [
            (
                torch.randn(2, 3, self._s, self._s),
                torch.randint(0, self._c, (2, self._s, self._s)),
            )
            for _ in range(self._n)
        ]

    def train_loader(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return self._batches()

    def val_loader(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return self._batches()


def _config(tmp_path: Path, overrides: dict | None = None) -> SegmentationConfig:
    raw: dict = {
        "name": "seg_run",
        "model": {"name": "unet", "num_classes": 3, "pretrained": False},
        "data": {"base_dir": str(tmp_path)},
        "training": {
            "learning_rate": 0.01,
            "epochs": 3,
            "batch_size": 2,
            "early_stopping_patience": 2,
            "seed": 0,
        },
        "output": {"models_dir": str(tmp_path / "models")},
        "device": {"kind": "cpu"},
    }
    if overrides:
        for key, val in overrides.items():
            raw.setdefault(key, {})
            if isinstance(val, dict):
                raw[key].update(val)
            else:
                raw[key] = val
    return SegmentationConfig.model_validate(raw)


# ── SegmentationMetricAccumulator ─────────────────────────────────────────────


class TestMetricAccumulator:
    def test_perfect_prediction(self) -> None:
        acc = SegmentationMetricAccumulator(num_classes=3, ignore_index=255)
        target = torch.tensor([[0, 1, 2], [2, 1, 0]])
        # logits arg-maxing exactly to target
        logits = torch.full((1, 3, 2, 3), -10.0)
        for r in range(2):
            for c in range(3):
                logits[0, target[r, c], r, c] = 10.0
        acc.update(logits, target.unsqueeze(0))
        miou, dice, pixel_acc = acc.compute()
        assert miou == 1.0
        assert dice == 1.0
        assert pixel_acc == 1.0

    def test_ignore_index_excluded(self) -> None:
        acc = SegmentationMetricAccumulator(num_classes=2, ignore_index=255)
        target = torch.tensor([[0, 255], [1, 255]]).unsqueeze(0)
        # predict class 0 everywhere; ignored pixels must not count against us
        logits = torch.zeros(1, 2, 2, 2)
        logits[0, 0] = 10.0  # always predict 0
        acc.update(logits, target)
        _, _, pixel_acc = acc.compute()
        # only (0,0)=0 correct and (1,0)=1 wrong among non-ignored -> 1/2
        assert pixel_acc == 0.5

    def test_half_correct_pixel_accuracy(self) -> None:
        acc = SegmentationMetricAccumulator(num_classes=2, ignore_index=255)
        target = torch.tensor([[0, 0], [1, 1]]).unsqueeze(0)
        logits = torch.zeros(1, 2, 2, 2)
        logits[0, 0] = 10.0  # predict 0 everywhere -> 2/4 correct
        acc.update(logits, target)
        _, _, pixel_acc = acc.compute()
        assert pixel_acc == 0.5


# ── dice_loss ─────────────────────────────────────────────────────────────────


class TestDiceLoss:
    def test_perfect_prediction_low_loss(self) -> None:
        target = torch.tensor([[0, 1, 2], [2, 1, 0]]).unsqueeze(0)
        logits = torch.full((1, 3, 2, 3), -10.0)
        for r in range(2):
            for c in range(3):
                logits[0, target[0, r, c], r, c] = 10.0
        loss = dice_loss(logits, target, ignore_index=255)
        assert loss.item() < 0.05

    def test_wrong_prediction_high_loss(self) -> None:
        target = torch.zeros(1, 4, 4, dtype=torch.long)
        logits = torch.full((1, 2, 4, 4), -10.0)
        logits[0, 1] = 10.0  # predict class 1 everywhere, target all 0
        loss = dice_loss(logits, target, ignore_index=255)
        assert loss.item() > 0.5

    def test_ignore_index_does_not_crash(self) -> None:
        target = torch.full((1, 4, 4), 255, dtype=torch.long)
        target[0, 0, 0] = 1
        logits = torch.randn(1, 2, 4, 4)
        loss = dice_loss(logits, target, ignore_index=255)
        assert torch.isfinite(loss)


# ── fit ───────────────────────────────────────────────────────────────────────


class TestFit:
    def test_runs_and_returns_result(self, tmp_path: Path) -> None:
        trainer = SegmentationTrainer(_config(tmp_path))
        result = trainer.fit(TinySegModel(3), FakeSegDataModule())
        assert isinstance(result, type(result))
        assert 1 <= result.best_epoch <= 3
        assert result.total_epochs <= 3
        assert result.model_path.is_file()

    def test_emits_sse_events(self, tmp_path: Path) -> None:
        events: list[dict] = []
        trainer = SegmentationTrainer(_config(tmp_path))
        trainer.fit(
            TinySegModel(3), FakeSegDataModule(), progress_callback=events.append
        )
        kinds = [e["event"] for e in events]
        assert kinds[0] == "start"
        assert kinds[-1] == "end"
        assert "epoch_end" in kinds
        epoch_ev = next(e for e in events if e["event"] == "epoch_end")
        assert "val_miou" in epoch_ev
        assert "val_loss" in epoch_ev

    def test_writes_run_json_with_seg_metrics(self, tmp_path: Path) -> None:
        trainer = SegmentationTrainer(_config(tmp_path))
        result = trainer.fit(TinySegModel(3), FakeSegDataModule())
        run_json = json.loads(
            (result.model_path.parent / "run.json").read_text(encoding="utf-8")
        )
        assert run_json["status"] == "completed"
        for key in ("miou", "dice", "pixel_acc", "best_epoch"):
            assert key in run_json["metrics"]
        assert "torch" in run_json["environment"]

    def test_combined_loss_runs(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path, {"training": {"loss": "combined"}})
        trainer = SegmentationTrainer(cfg)
        result = trainer.fit(TinySegModel(3), FakeSegDataModule())
        assert result.model_path.is_file()

    def test_dice_loss_mode_runs(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path, {"training": {"loss": "dice"}})
        trainer = SegmentationTrainer(cfg)
        result = trainer.fit(TinySegModel(3), FakeSegDataModule())
        assert result.model_path.is_file()

    def test_evaluate_returns_metrics(self, tmp_path: Path) -> None:
        trainer = SegmentationTrainer(_config(tmp_path))
        miou, dice, pixel_acc = trainer.evaluate(
            TinySegModel(3), FakeSegDataModule().val_loader()
        )
        assert 0.0 <= miou <= 1.0
        assert 0.0 <= dice <= 1.0
        assert 0.0 <= pixel_acc <= 1.0
