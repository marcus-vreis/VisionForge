import json
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

from visionforge.core.trainer import (
    EpochResult,
    Trainer,
    TrainResult,
    _seed_everything,
)
from visionforge.utils.config import ExperimentConfig

# ── helpers ───────────────────────────────────────────────────────────────────


class DummyBinaryModel(nn.Module):
    """Minimal model that accepts (B, 3, 32, 32) and returns (B, 1)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3 * 32 * 32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(1))


class FakeDataModule:
    """Provides tiny fake batches for trainer tests."""

    def __init__(self, n_batches: int = 2) -> None:
        self._n = n_batches

    def _batches(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [
            (torch.randn(4, 3, 32, 32), torch.randint(0, 2, (4,)))
            for _ in range(self._n)
        ]

    def train_loader(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return self._batches()

    def val_loader(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return self._batches()


@pytest.fixture
def minimal_config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "name": "test_run",
            "task": "binary",
            "model": {"name": "resnet18", "num_classes": 1, "pretrained": False},
            "training": {
                "learning_rate": 0.1,
                "epochs": 3,
                "batch_size": 4,
                "early_stopping_patience": 2,
                "seed": 0,
            },
            "data": {"base_dir": str(tmp_path)},
            "output": {
                "models_dir": str(tmp_path / "models"),
                "graphics_dir": str(tmp_path / "graphics"),
                "logs_dir": str(tmp_path / "logs"),
                "reports_dir": str(tmp_path / "reports"),
            },
        }
    )


# ── unit tests ────────────────────────────────────────────────────────────────


class TestSeedEverything:
    def test_does_not_raise(self) -> None:
        """_seed_everything() must complete without exceptions."""
        _seed_everything(42)
        _seed_everything(0)

    def test_reproducible_tensors(self) -> None:
        """Same seed must produce identical random tensors."""
        _seed_everything(7)
        a = torch.randn(10)
        _seed_everything(7)
        b = torch.randn(10)
        assert torch.allclose(a, b)


class TestDataclasses:
    def test_epoch_result(self) -> None:
        """EpochResult must store all four fields."""
        r = EpochResult(epoch=1, train_loss=0.5, val_loss=0.4, val_accuracy=0.8)
        assert r.epoch == 1
        assert r.train_loss == 0.5

    def test_train_result(self) -> None:
        """TrainResult must track best_epoch and history."""
        r = TrainResult(best_epoch=3, best_val_loss=0.2, total_epochs=5)
        assert r.best_epoch == 3
        assert r.history == []


class TestTrainerFit:
    def test_fit_completes(self, minimal_config: ExperimentConfig) -> None:
        """fit() must return a TrainResult without raising."""
        trainer = Trainer(minimal_config)
        result = trainer.fit(DummyBinaryModel(), FakeDataModule())
        assert isinstance(result, TrainResult)

    def test_run_json_written(self, minimal_config: ExperimentConfig) -> None:
        """fit() must write a run.json file with required keys."""
        trainer = Trainer(minimal_config)
        trainer.fit(DummyBinaryModel(), FakeDataModule())

        run_dirs = list(
            (minimal_config.output.models_dir / minimal_config.name).glob("*/run.json")
        )
        assert len(run_dirs) == 1

        data: dict[str, Any] = json.loads(run_dirs[0].read_text(encoding="utf-8"))
        assert "id" in data
        assert "experiment" in data
        assert "status" in data
        assert "metrics" in data
        assert "history" in data
        assert "artifacts" in data

    def test_model_checkpoint_saved(self, minimal_config: ExperimentConfig) -> None:
        """fit() must save best_model.pth."""
        trainer = Trainer(minimal_config)
        result = trainer.fit(DummyBinaryModel(), FakeDataModule())
        assert result.model_path.exists()

    def test_early_stopping_triggers(
        self, minimal_config: ExperimentConfig, tmp_path: Path
    ) -> None:
        """fit() must stop before max epochs when val_loss does not improve."""
        # patience=2, epochs=10 — early stopping should kick in well before epoch 10
        config = ExperimentConfig.model_validate(
            {
                **minimal_config.model_dump(mode="json"),
                "training": {
                    **minimal_config.training.model_dump(mode="json"),
                    "epochs": 10,
                    "early_stopping_patience": 2,
                },
            }
        )
        trainer = Trainer(config)

        # bias is required so the optimizer doesn't get an empty parameter list;
        # the model returns a constant, so val_loss never improves.
        class ConstantModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.bias = nn.Parameter(torch.zeros(1))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.zeros(x.size(0), 1) + self.bias * 0

        result = trainer.fit(ConstantModel(), FakeDataModule())
        assert result.total_epochs < 10

    def test_history_entries_match_epochs_trained(
        self, minimal_config: ExperimentConfig
    ) -> None:
        """History list must have one entry per epoch actually trained."""
        trainer = Trainer(minimal_config)
        result = trainer.fit(DummyBinaryModel(), FakeDataModule())
        assert len(result.history) == result.total_epochs


class DummyMulticlassModel(nn.Module):
    """Minimal model that accepts (B, 3, 32, 32) and returns (B, 3)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3 * 32 * 32, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(1))


class FakeMulticlassDataModule:
    """Provides tiny fake batches with 3 classes."""

    def _batches(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [
            (torch.randn(6, 3, 32, 32), torch.randint(0, 3, (6,))) for _ in range(2)
        ]

    def train_loader(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return self._batches()

    def val_loader(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return self._batches()


class TestTrainerMulticlass:
    @pytest.fixture
    def multiclass_config(self, tmp_path: Path) -> ExperimentConfig:
        return ExperimentConfig.model_validate(
            {
                "name": "mc_run",
                "task": "multiclass",
                "model": {"name": "resnet18", "num_classes": 3, "pretrained": False},
                "training": {
                    "learning_rate": 0.1,
                    "epochs": 2,
                    "batch_size": 4,
                    "early_stopping_patience": 5,
                    "seed": 0,
                },
                "data": {"base_dir": str(tmp_path)},
                "output": {
                    "models_dir": str(tmp_path / "models"),
                    "graphics_dir": str(tmp_path / "graphics"),
                    "logs_dir": str(tmp_path / "logs"),
                    "reports_dir": str(tmp_path / "reports"),
                },
            }
        )

    def test_multiclass_fit_completes(
        self, multiclass_config: ExperimentConfig
    ) -> None:
        """fit() with multiclass config must return a TrainResult."""
        trainer = Trainer(multiclass_config)
        result = trainer.fit(DummyMulticlassModel(), FakeMulticlassDataModule())
        assert isinstance(result, TrainResult)
        assert result.model_path.exists()


class TestTrainerOptimizers:
    @pytest.fixture
    def base_config(self, tmp_path: Path) -> dict[str, Any]:
        return {
            "name": "opt_run",
            "task": "binary",
            "model": {"name": "resnet18", "num_classes": 1, "pretrained": False},
            "training": {
                "learning_rate": 0.1,
                "epochs": 1,
                "batch_size": 4,
                "early_stopping_patience": 5,
                "seed": 0,
            },
            "data": {"base_dir": str(tmp_path)},
            "output": {
                "models_dir": str(tmp_path / "models"),
                "graphics_dir": str(tmp_path / "graphics"),
                "logs_dir": str(tmp_path / "logs"),
                "reports_dir": str(tmp_path / "reports"),
            },
        }

    @pytest.mark.parametrize("optimizer", ["adam", "sgd", "adamw"])
    def test_all_optimizers_complete(
        self, base_config: dict[str, Any], optimizer: str
    ) -> None:
        """fit() must complete with each supported optimizer."""
        base_config["training"]["optimizer"] = optimizer
        config = ExperimentConfig.model_validate(base_config)
        trainer = Trainer(config)
        result = trainer.fit(DummyBinaryModel(), FakeDataModule())
        assert isinstance(result, TrainResult)
