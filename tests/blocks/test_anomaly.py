from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch

from visionforge.blocks.anomaly import AnomalyBlock
from visionforge.models.anomaly_factory import ConvAutoencoder, PatchCore
from visionforge.utils.anomaly_config import AnomalyConfig


class FakeDataModule:
    closed = False

    def __init__(self, config: AnomalyConfig) -> None:
        pass

    def _batches(self, labels: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [(torch.randn(2, 3, 32, 32), labels) for _ in range(2)]

    def train_loader(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return self._batches(torch.zeros(2, dtype=torch.long))

    def test_loader(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return self._batches(torch.tensor([0, 1]))

    def close(self) -> None:
        """The blocks shut their data module down in a finally; record it."""
        self.closed = True


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


class _NormalOnlyDataModule(FakeDataModule):
    def test_loader(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return self._batches(torch.zeros(2, dtype=torch.long))


class TestOneClassTestSplit:
    """PatchCore on CPU, 32x32 synthetic images, test/ with normals only (#170)."""

    def _run(self, tmp_path: Path) -> dict[str, Any]:
        config = AnomalyConfig.model_validate(
            {
                "name": "anom_one_class",
                "model": {
                    "name": "patchcore",
                    "backbone": "resnet18",
                    "pretrained": False,
                    "coreset_ratio": 0.5,
                },
                "data": {"base_dir": str(tmp_path), "image_size": 32},
                "training": {"batch_size": 2},
                "output": {"models_dir": str(tmp_path / "models")},
                "device": {"kind": "cpu"},
            }
        )
        block = AnomalyBlock()
        block.setup(config)
        with (
            patch(
                "visionforge.blocks.anomaly.AnomalyModelFactory.create",
                return_value=PatchCore(
                    backbone="resnet18", pretrained=False, coreset_ratio=0.5
                ),
            ),
            patch(
                "visionforge.blocks.anomaly.AnomalyDataModule",
                lambda cfg: _NormalOnlyDataModule(cfg),
            ),
        ):
            block.run()
        self.report = block.report()
        run_dir = Path(self.report["train"]["run_dir"])
        data: dict[str, Any] = json.loads(
            (run_dir / "run.json").read_text(encoding="utf-8")
        )
        return data

    def test_test_auroc_is_null(self, tmp_path: Path) -> None:
        data = self._run(tmp_path)

        assert data["metrics"]["test_auroc"] is None
        assert data["metrics"]["test_image_f1"] is None
        assert self.report["test"]["auroc"] is None

    def test_run_carries_the_warning(self, tmp_path: Path) -> None:
        data = self._run(tmp_path)

        codes = [w["code"] for w in data["warnings"]]
        assert codes.count("single_class_test_split") == 1
