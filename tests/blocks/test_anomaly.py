from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import torch

from visionforge.blocks.anomaly import AnomalyBlock
from visionforge.models.anomaly_factory import ConvAutoencoder
from visionforge.utils.anomaly_config import AnomalyConfig


class FakeDataModule:
    def __init__(self, config: AnomalyConfig) -> None:
        pass

    def _batches(self, labels: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [(torch.randn(2, 3, 32, 32), labels) for _ in range(2)]

    def train_loader(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return self._batches(torch.zeros(2, dtype=torch.long))

    def test_loader(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return self._batches(torch.tensor([0, 1]))


def _config(tmp_path: Path) -> AnomalyConfig:
    return AnomalyConfig.model_validate(
        {
            "name": "anom_block",
            "model": {"name": "autoencoder", "latent_dim": 8},
            "data": {"base_dir": str(tmp_path), "image_size": 32},
            "training": {"epochs": 2, "batch_size": 2, "early_stopping_patience": 2},
            "output": {"models_dir": str(tmp_path / "models")},
            "device": {"kind": "cpu"},
        }
    )


def _run_block(config: AnomalyConfig) -> AnomalyBlock:
    block = AnomalyBlock()
    block.setup(config)
    with (
        patch(
            "visionforge.blocks.anomaly.AnomalyModelFactory.create",
            return_value=ConvAutoencoder(latent_dim=8),
        ),
        patch(
            "visionforge.blocks.anomaly.AnomalyDataModule",
            lambda cfg: FakeDataModule(cfg),
        ),
    ):
        block.run()
    return block


class TestAnomalyBlock:
    def test_run_produces_train_and_test_report(self, tmp_path: Path) -> None:
        block = _run_block(_config(tmp_path))
        report = block.report()
        assert "train" in report
        assert report["train"]["total_epochs"] >= 1
        assert "test" in report
        assert set(report["test"]) == {"auroc", "threshold", "image_f1"}

    def test_run_writes_plot_and_test_metrics(self, tmp_path: Path) -> None:
        block = _run_block(_config(tmp_path))
        run_dir = Path(block.report()["train"]["run_dir"])
        assert (run_dir / "auroc.png").is_file()
        data = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        assert "test_auroc" in data["metrics"]
        # ADR-077: the score histogram shows where the two populations overlap
        # and what the chosen threshold keeps.
        graphics = [Path(p).name for p in data["artifacts"]["graphics"]]
        assert graphics == ["auroc.png", "score_histogram.png"]
        assert all((run_dir / name).is_file() for name in graphics)

    def test_report_empty_before_run(self, tmp_path: Path) -> None:
        block = AnomalyBlock()
        block.setup(_config(tmp_path))
        assert block.report() == {}
