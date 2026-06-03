from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from visionforge.core.regression_trainer import (
    RegressionTrainer,
    RegressionTrainResult,
    _MetricAccumulator,
)
from visionforge.utils.regression_config import RegressionConfig


class DummyRegressor(nn.Module):
    """Minimal model: (B, 3, 32, 32) -> (B, num_targets)."""

    def __init__(self, num_targets: int = 1) -> None:
        super().__init__()
        self.fc = nn.Linear(3 * 32 * 32, num_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(1))


class FakeRegressionDataModule:
    """Tiny fixed batches of (image, float target) for trainer tests."""

    def __init__(self, num_targets: int = 1, n_batches: int = 2) -> None:
        self._t = num_targets
        self._n = n_batches

    def _batches(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [
            (torch.randn(4, 3, 32, 32), torch.randn(4, self._t)) for _ in range(self._n)
        ]

    def train_loader(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return self._batches()

    def val_loader(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return self._batches()


def _config(tmp_path: Path, overrides: dict | None = None) -> RegressionConfig:
    raw: dict = {
        "name": "reg_run",
        "model": {"name": "resnet18", "num_targets": 1, "pretrained": False},
        "data": {"base_dir": str(tmp_path), "target_columns": ["target"]},
        "training": {
            "learning_rate": 0.01,
            "epochs": 3,
            "batch_size": 4,
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
    return RegressionConfig.model_validate(raw)


# ── _MetricAccumulator ────────────────────────────────────────────────────────


class TestMetricAccumulator:
    def test_perfect_prediction(self) -> None:
        acc = _MetricAccumulator()
        y = torch.tensor([[1.0], [2.0], [3.0]])
        acc.update(y.clone(), y)
        mse, rmse, mae, r2 = acc.compute()
        assert mse == pytest.approx(0.0)
        assert rmse == pytest.approx(0.0)
        assert mae == pytest.approx(0.0)
        assert r2 == pytest.approx(1.0)

    def test_known_error(self) -> None:
        acc = _MetricAccumulator()
        targets = torch.tensor([[0.0], [2.0], [4.0]])
        preds = torch.tensor([[1.0], [1.0], [5.0]])  # errors: +1, -1, +1
        acc.update(preds, targets)
        mse, rmse, mae, r2 = acc.compute()
        assert mse == pytest.approx(1.0)  # (1+1+1)/3
        assert rmse == pytest.approx(1.0)
        assert mae == pytest.approx(1.0)
        # ss_tot = var*n = ((0-2)^2+(2-2)^2+(4-2)^2) = 8 ; r2 = 1 - 3/8 = 0.625
        assert r2 == pytest.approx(0.625)

    def test_empty_is_zero(self) -> None:
        assert _MetricAccumulator().compute() == (0.0, 0.0, 0.0, 0.0)

    def test_constant_target_r2_zero(self) -> None:
        acc = _MetricAccumulator()
        targets = torch.tensor([[2.0], [2.0]])
        preds = torch.tensor([[2.1], [1.9]])
        acc.update(preds, targets)
        _, _, _, r2 = acc.compute()
        assert r2 == 0.0  # ss_tot == 0 guard


# ── RegressionTrainer ─────────────────────────────────────────────────────────


class TestRegressionTrainer:
    def test_fit_returns_result_and_checkpoint(self, tmp_path: Path) -> None:
        trainer = RegressionTrainer(_config(tmp_path))
        result = trainer.fit(DummyRegressor(), FakeRegressionDataModule())
        assert isinstance(result, RegressionTrainResult)
        assert result.total_epochs >= 1
        assert result.best_epoch >= 1
        assert result.model_path.is_file()

    def test_writes_run_json_with_regression_metrics(self, tmp_path: Path) -> None:
        trainer = RegressionTrainer(_config(tmp_path))
        result = trainer.fit(DummyRegressor(), FakeRegressionDataModule())
        run_json = result.model_path.parent / "run.json"
        data = json.loads(run_json.read_text(encoding="utf-8"))
        assert data["config"]["task"] == "regression"
        for key in ("mse", "rmse", "mae", "r2", "best_val_loss"):
            assert key in data["metrics"]
        assert data["history"][0]["val_rmse"] is not None

    def test_progress_callback_emits_events(self, tmp_path: Path) -> None:
        events: list[str] = []
        trainer = RegressionTrainer(_config(tmp_path))
        trainer.fit(
            DummyRegressor(),
            FakeRegressionDataModule(),
            progress_callback=lambda p: events.append(p["event"]),
        )
        assert events[0] == "start"
        assert events[-1] == "end"
        assert "epoch_end" in events

    def test_multi_target(self, tmp_path: Path) -> None:
        cfg = _config(
            tmp_path,
            {
                "model": {"name": "resnet18", "num_targets": 2, "pretrained": False},
                "data": {"base_dir": str(tmp_path), "target_columns": ["a", "b"]},
            },
        )
        trainer = RegressionTrainer(cfg)
        result = trainer.fit(DummyRegressor(num_targets=2), FakeRegressionDataModule(2))
        assert result.model_path.is_file()

    def test_mae_and_huber_losses(self, tmp_path: Path) -> None:
        for loss in ("mae", "huber"):
            base = tmp_path / loss
            base.mkdir(exist_ok=True)
            cfg = _config(base, {"training": {"loss": loss}})
            trainer = RegressionTrainer(cfg)
            result = trainer.fit(DummyRegressor(), FakeRegressionDataModule())
            assert result.model_path.is_file()

    def test_device_label_cpu(self, tmp_path: Path) -> None:
        trainer = RegressionTrainer(_config(tmp_path))
        assert "cpu" in trainer.device_label.lower()
