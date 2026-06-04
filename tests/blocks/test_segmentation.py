from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn

from visionforge.blocks.segmentation import SegmentationBlock
from visionforge.utils.segmentation_config import SegmentationConfig


class TinySegModel(nn.Module):
    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class FakeDataModule:
    def __init__(self, config: SegmentationConfig, *, with_test: bool = True) -> None:
        self._with_test = with_test

    def _batches(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [
            (torch.randn(2, 3, 16, 16), torch.randint(0, 3, (2, 16, 16)))
            for _ in range(2)
        ]

    def train_loader(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return self._batches()

    def val_loader(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return self._batches()

    def test_loader(self) -> list[tuple[torch.Tensor, torch.Tensor]] | None:
        return self._batches() if self._with_test else None


def _config(tmp_path: Path) -> SegmentationConfig:
    return SegmentationConfig.model_validate(
        {
            "name": "seg_block",
            "model": {"name": "unet", "num_classes": 3, "pretrained": False},
            "data": {"base_dir": str(tmp_path)},
            "training": {"epochs": 2, "batch_size": 2, "early_stopping_patience": 2},
            "output": {"models_dir": str(tmp_path / "models")},
            "device": {"kind": "cpu"},
        }
    )


def _run_block(
    config: SegmentationConfig, *, with_test: bool = True
) -> SegmentationBlock:
    block = SegmentationBlock()
    block.setup(config)
    with (
        patch(
            "visionforge.blocks.segmentation.SegmentationModelFactory.create",
            return_value=TinySegModel(),
        ),
        patch(
            "visionforge.blocks.segmentation.SegmentationDataModule",
            lambda cfg: FakeDataModule(cfg, with_test=with_test),
        ),
    ):
        block.run()
    return block


class TestSegmentationBlock:
    def test_run_produces_train_and_test_report(self, tmp_path: Path) -> None:
        block = _run_block(_config(tmp_path))
        report = block.report()
        assert "train" in report
        assert report["train"]["total_epochs"] >= 1
        assert "test" in report
        assert set(report["test"]) == {"miou", "dice", "pixel_acc"}

    def test_run_writes_loss_plot_and_test_metrics(self, tmp_path: Path) -> None:
        block = _run_block(_config(tmp_path))
        run_dir = Path(block.report()["train"]["run_dir"])
        assert (run_dir / "loss.png").is_file()
        data = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        assert "test_miou" in data["metrics"]
        assert data["artifacts"]["graphics"] == [str(run_dir / "loss.png")]

    def test_no_test_split_skips_test_metrics(self, tmp_path: Path) -> None:
        block = _run_block(_config(tmp_path), with_test=False)
        report = block.report()
        assert "train" in report
        assert "test" not in report

    def test_report_empty_before_run(self, tmp_path: Path) -> None:
        block = SegmentationBlock()
        block.setup(_config(tmp_path))
        assert block.report() == {}
