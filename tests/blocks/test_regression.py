from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn

from visionforge.blocks.regression import RegressionBlock
from visionforge.utils.regression_config import RegressionConfig


class DummyRegressor(nn.Module):
    def __init__(self, num_targets: int = 1) -> None:
        super().__init__()
        self.fc = nn.Linear(3 * 32 * 32, num_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(1))


class FakeDataModule:
    def __init__(self, config: RegressionConfig, *, with_test: bool = True) -> None:
        self._with_test = with_test

    def _batches(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [(torch.randn(4, 3, 32, 32), torch.randn(4, 1)) for _ in range(2)]

    def train_loader(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return self._batches()

    def val_loader(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return self._batches()

    def test_loader(self) -> list[tuple[torch.Tensor, torch.Tensor]] | None:
        return self._batches() if self._with_test else None


def _config(tmp_path: Path) -> RegressionConfig:
    return RegressionConfig.model_validate(
        {
            "name": "reg_block",
            "model": {"name": "resnet18", "num_targets": 1, "pretrained": False},
            "data": {"base_dir": str(tmp_path), "target_columns": ["target"]},
            "training": {"epochs": 2, "batch_size": 4, "early_stopping_patience": 2},
            "output": {"models_dir": str(tmp_path / "models")},
            "device": {"kind": "cpu"},
        }
    )


def _run_block(config: RegressionConfig, *, with_test: bool = True) -> RegressionBlock:
    block = RegressionBlock()
    block.setup(config)
    with (
        patch(
            "visionforge.blocks.regression.RegressionModelFactory.create",
            return_value=DummyRegressor(),
        ),
        patch(
            "visionforge.blocks.regression.RegressionDataModule",
            lambda cfg: FakeDataModule(cfg, with_test=with_test),
        ),
    ):
        block.run()
    return block


class TestRegressionBlock:
    def test_run_produces_train_and_test_report(self, tmp_path: Path) -> None:
        block = _run_block(_config(tmp_path))
        report = block.report()
        assert "train" in report
        assert report["train"]["total_epochs"] >= 1
        assert "test" in report
        assert set(report["test"]) == {"mse", "rmse", "mae", "r2"}

    def test_run_writes_loss_plot_and_test_metrics(self, tmp_path: Path) -> None:
        block = _run_block(_config(tmp_path))
        run_dir = Path(block.report()["train"]["run_dir"])
        assert (run_dir / "loss.png").is_file()
        data = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        assert "test_rmse" in data["metrics"]
        # ADR-077: the loss curve plus the test-set diagnostics an aggregate
        # error cannot give — whether the model tracks the target at all.
        graphics = [Path(p).name for p in data["artifacts"]["graphics"]]
        assert graphics == ["loss.png", "pred_vs_true.png", "residuals.png"]
        assert all((run_dir / name).is_file() for name in graphics)

    def test_no_test_split_skips_test_metrics(self, tmp_path: Path) -> None:
        block = _run_block(_config(tmp_path), with_test=False)
        report = block.report()
        assert "train" in report
        assert "test" not in report

    def test_report_empty_before_run(self, tmp_path: Path) -> None:
        block = RegressionBlock()
        block.setup(_config(tmp_path))
        assert block.report() == {}
