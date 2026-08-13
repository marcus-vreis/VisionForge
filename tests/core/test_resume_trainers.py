"""Every task can pick up where it stopped, not only classification.

The unit tests in ``test_resume.py`` cover the state file itself. These drive the
real loops: stop a run at its first epoch, hand the directory back, and check the
second attempt continues the same run rather than starting a new one — same
directory, one continuous history, and the resume file gone once it finishes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from visionforge.core.anomaly_trainer import AnomalyTrainer
from visionforge.core.cancellation import CancellationToken
from visionforge.core.regression_trainer import RegressionTrainer
from visionforge.core.resume import RESUME_FILENAME, load_resume_state
from visionforge.core.segmentation_trainer import SegmentationTrainer
from visionforge.core.trainer import Trainer
from visionforge.models.anomaly_factory import ConvAutoencoder
from visionforge.utils.anomaly_config import AnomalyConfig
from visionforge.utils.config import ExperimentConfig
from visionforge.utils.regression_config import RegressionConfig
from visionforge.utils.segmentation_config import SegmentationConfig

EPOCHS = 3


def _stop_after_first_epoch(token: CancellationToken) -> Any:
    """Progress callback that cancels once epoch 1 has been recorded."""

    def _callback(event: dict[str, Any]) -> None:
        if event.get("event") == "epoch_end" and event.get("epoch") == 1:
            token.cancel()

    return _callback


def _run_dirs(models_dir: Path, name: str) -> list[Path]:
    parent = models_dir / name
    return sorted(p for p in parent.iterdir() if p.is_dir()) if parent.exists() else []


# ── models and data ───────────────────────────────────────────────────────────


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3 * 32 * 32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(1))


class TinyRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3 * 32 * 32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(1))


class TinySegModel(nn.Module):
    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class FakeClassificationData:
    def _batches(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [
            (torch.randn(4, 3, 32, 32), torch.randint(0, 2, (4,))) for _ in range(2)
        ]

    def train_loader(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return self._batches()

    def val_loader(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return self._batches()


class FakeRegressionData:
    def _batches(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [(torch.randn(4, 3, 32, 32), torch.randn(4, 1)) for _ in range(2)]

    def train_loader(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return self._batches()

    def val_loader(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return self._batches()


class FakeSegData:
    def _batches(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [
            (torch.randn(2, 3, 16, 16), torch.randint(0, 3, (2, 16, 16)))
            for _ in range(2)
        ]

    def train_loader(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return self._batches()

    def val_loader(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return self._batches()


class FakeAnomalyData:
    def train_loader(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [
            (torch.randn(2, 3, 32, 32), torch.zeros(2, dtype=torch.long))
            for _ in range(2)
        ]

    def test_loader(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [(torch.randn(2, 3, 32, 32), torch.tensor([0, 1])) for _ in range(2)]


# ── configs ───────────────────────────────────────────────────────────────────


def _training(epochs: int = EPOCHS, batch: int = 4) -> dict[str, Any]:
    return {
        "learning_rate": 0.01,
        "epochs": epochs,
        "batch_size": batch,
        "early_stopping_patience": 99,  # never stop early: the test owns the stop
        "seed": 0,
    }


def _classification_config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "name": "cls_resume",
            "task": "binary",
            "model": {"name": "resnet18", "num_classes": 1, "pretrained": False},
            "training": _training(),
            "data": {"base_dir": str(tmp_path)},
            "output": {"models_dir": str(tmp_path / "models")},
            "device": {"kind": "cpu"},
        }
    )


def _regression_config(tmp_path: Path) -> RegressionConfig:
    return RegressionConfig.model_validate(
        {
            "name": "reg_resume",
            "model": {"name": "resnet18", "num_targets": 1, "pretrained": False},
            "data": {"base_dir": str(tmp_path), "target_columns": ["target"]},
            "training": _training(),
            "output": {"models_dir": str(tmp_path / "models")},
            "device": {"kind": "cpu"},
        }
    )


def _segmentation_config(tmp_path: Path) -> SegmentationConfig:
    return SegmentationConfig.model_validate(
        {
            "name": "seg_resume",
            "model": {"name": "unet", "num_classes": 3, "pretrained": False},
            "data": {"base_dir": str(tmp_path)},
            "training": _training(batch=2),
            "output": {"models_dir": str(tmp_path / "models")},
            "device": {"kind": "cpu"},
        }
    )


def _anomaly_config(tmp_path: Path) -> AnomalyConfig:
    return AnomalyConfig.model_validate(
        {
            "name": "anom_resume",
            "model": {"name": "autoencoder", "latent_dim": 8},
            "data": {"base_dir": str(tmp_path), "image_size": 32},
            "training": _training(batch=2),
            "output": {"models_dir": str(tmp_path / "models")},
            "device": {"kind": "cpu"},
        }
    )


# ── the four loops ────────────────────────────────────────────────────────────


class TestClassificationResume:
    def test_continues_the_same_run(self, tmp_path: Path) -> None:
        cfg = _classification_config(tmp_path)
        token = CancellationToken()

        stopped = Trainer(cfg).fit(
            TinyClassifier(),
            FakeClassificationData(),
            progress_callback=_stop_after_first_epoch(token),
            cancel_token=token,
        )
        run_dir = stopped.model_path.parent
        assert stopped.total_epochs == 1
        assert (run_dir / RESUME_FILENAME).is_file()

        finished = Trainer(cfg).fit(
            TinyClassifier(), FakeClassificationData(), resume_dir=run_dir
        )

        assert finished.model_path.parent == run_dir
        assert [h.epoch for h in finished.history] == [1, 2, 3]
        assert _run_dirs(tmp_path / "models", "cls_resume") == [run_dir]
        assert not (run_dir / RESUME_FILENAME).exists()

    def test_the_stopped_state_records_the_epoch_reached(self, tmp_path: Path) -> None:
        cfg = _classification_config(tmp_path)
        token = CancellationToken()

        stopped = Trainer(cfg).fit(
            TinyClassifier(),
            FakeClassificationData(),
            progress_callback=_stop_after_first_epoch(token),
            cancel_token=token,
        )

        state = load_resume_state(stopped.model_path.parent)
        assert state is not None
        assert state.epoch == 1
        assert len(state.history) == 1

    def test_unreadable_state_starts_a_fresh_run(self, tmp_path: Path) -> None:
        """A corrupt file must not hijack the old directory and overwrite it."""
        cfg = _classification_config(tmp_path)
        token = CancellationToken()
        stopped = Trainer(cfg).fit(
            TinyClassifier(),
            FakeClassificationData(),
            progress_callback=_stop_after_first_epoch(token),
            cancel_token=token,
        )
        run_dir = stopped.model_path.parent
        (run_dir / RESUME_FILENAME).write_bytes(b"not a checkpoint")

        finished = Trainer(cfg).fit(
            TinyClassifier(), FakeClassificationData(), resume_dir=run_dir
        )

        assert finished.model_path.parent != run_dir
        assert [h.epoch for h in finished.history] == [1, 2, 3]


class TestRegressionResume:
    def test_continues_the_same_run(self, tmp_path: Path) -> None:
        cfg = _regression_config(tmp_path)
        token = CancellationToken()

        stopped = RegressionTrainer(cfg).fit(
            TinyRegressor(),
            FakeRegressionData(),
            progress_callback=_stop_after_first_epoch(token),
            cancel_token=token,
        )
        run_dir = stopped.model_path.parent
        assert stopped.total_epochs == 1
        assert (run_dir / RESUME_FILENAME).is_file()

        finished = RegressionTrainer(cfg).fit(
            TinyRegressor(), FakeRegressionData(), resume_dir=run_dir
        )

        assert finished.model_path.parent == run_dir
        assert [h.epoch for h in finished.history] == [1, 2, 3]
        assert _run_dirs(tmp_path / "models", "reg_resume") == [run_dir]
        assert not (run_dir / RESUME_FILENAME).exists()


class TestSegmentationResume:
    def test_continues_the_same_run(self, tmp_path: Path) -> None:
        cfg = _segmentation_config(tmp_path)
        token = CancellationToken()

        stopped = SegmentationTrainer(cfg).fit(
            TinySegModel(),
            FakeSegData(),
            progress_callback=_stop_after_first_epoch(token),
            cancel_token=token,
        )
        run_dir = stopped.model_path.parent
        assert stopped.total_epochs == 1
        assert (run_dir / RESUME_FILENAME).is_file()

        finished = SegmentationTrainer(cfg).fit(
            TinySegModel(), FakeSegData(), resume_dir=run_dir
        )

        assert finished.model_path.parent == run_dir
        assert [h.epoch for h in finished.history] == [1, 2, 3]
        assert _run_dirs(tmp_path / "models", "seg_resume") == [run_dir]
        assert not (run_dir / RESUME_FILENAME).exists()


class TestAnomalyResume:
    def test_continues_the_same_run(self, tmp_path: Path) -> None:
        cfg = _anomaly_config(tmp_path)
        token = CancellationToken()

        stopped = AnomalyTrainer(cfg).fit(
            ConvAutoencoder(latent_dim=8),
            FakeAnomalyData(),
            progress_callback=_stop_after_first_epoch(token),
            cancel_token=token,
        )
        run_dir = stopped.model_path.parent
        assert stopped.total_epochs == 1
        assert (run_dir / RESUME_FILENAME).is_file()

        finished = AnomalyTrainer(cfg).fit(
            ConvAutoencoder(latent_dim=8), FakeAnomalyData(), resume_dir=run_dir
        )

        assert finished.model_path.parent == run_dir
        assert [h.epoch for h in finished.history] == [1, 2, 3]
        assert _run_dirs(tmp_path / "models", "anom_resume") == [run_dir]
        assert not (run_dir / RESUME_FILENAME).exists()
