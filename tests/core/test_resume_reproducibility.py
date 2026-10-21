"""Epoch-boundary resume preserves the stochastic CPU training trajectory."""

from __future__ import annotations

import copy
import json
import random
from collections.abc import Callable
from functools import partial
from pathlib import Path
from threading import Event
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from visionforge.core.anomaly_trainer import AnomalyTrainer
from visionforge.core.cancellation import CancellationToken
from visionforge.core.detection_trainer import DetectionTrainer
from visionforge.core.regression_trainer import RegressionTrainer
from visionforge.core.resume import load_resume_state
from visionforge.core.segmentation_trainer import SegmentationTrainer
from visionforge.core.trainer import Trainer
from visionforge.models.anomaly_factory import ConvAutoencoder
from visionforge.utils.anomaly_config import AnomalyConfig
from visionforge.utils.config import ExperimentConfig
from visionforge.utils.detection_config import DetectionConfig
from visionforge.utils.regression_config import RegressionConfig
from visionforge.utils.segmentation_config import SegmentationConfig


class RandomizedDataset(Dataset):
    """Small fixed examples with augmentation from all three CPU RNGs."""

    def __len__(self) -> int:
        return 12

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        value = torch.arange(4, dtype=torch.float32) / 10 + index / 12
        value = value + random.random() / 10 + float(np.random.random()) / 10
        return value + torch.rand(4) / 10, torch.tensor(index % 2)


class RandomizedData:
    """Construct loaders that deliberately consume RNG during setup."""

    def __init__(self, explicit_generator: bool = False) -> None:
        self.explicit_generator = explicit_generator

    def train_loader(self) -> DataLoader:
        """Return shuffled CPU batches."""
        torch.rand(3)
        generator = (
            torch.Generator().manual_seed(37) if self.explicit_generator else None
        )
        return DataLoader(
            RandomizedDataset(),
            batch_size=4,
            shuffle=True,
            num_workers=0,
            generator=generator,
        )

    def val_loader(self) -> DataLoader:
        """Return evaluation batches."""
        return DataLoader(RandomizedDataset(), batch_size=4, num_workers=0)


def _config(path: Path) -> ExperimentConfig:
    (path / "dataset").mkdir(parents=True, exist_ok=True)
    return ExperimentConfig.model_validate(
        {
            "name": "trajectory",
            "task": "binary",
            "model": {"name": "resnet18", "num_classes": 1, "pretrained": False},
            "training": {"epochs": 3, "seed": 19, "early_stopping_patience": 0},
            "data": {"base_dir": str(path / "dataset")},
            "output": {"models_dir": str(path / "models")},
            "device": {"kind": "cpu"},
        }
    )


@pytest.mark.parametrize("explicit_generator", [False, True])
@pytest.mark.parametrize("stop_epoch", [1, 2])
def test_stochastic_resume_matches_continuous_training(
    tmp_path: Path, explicit_generator: bool, stop_epoch: int
) -> None:
    """Resumed history and final tensors must be bitwise equal on CPU."""
    torch.manual_seed(7)
    initial = nn.Sequential(nn.Dropout(0.4), nn.Linear(4, 1))
    continuous_model = copy.deepcopy(initial)
    continuous_cfg = _config(tmp_path / "continuous")
    continuous_cfg.training.scheduler.kind = "step"
    continuous_cfg.training.scheduler.step_size = 1
    continuous = Trainer(continuous_cfg).fit(
        continuous_model, RandomizedData(explicit_generator)
    )
    token = CancellationToken()

    def stop(event: dict[str, Any]) -> None:
        if event.get("event") == "epoch_end" and event.get("epoch") == stop_epoch:
            token.cancel()

    cfg = _config(tmp_path / "resumed")
    cfg.training.scheduler = continuous_cfg.training.scheduler.model_copy()
    stopped = Trainer(cfg).fit(
        copy.deepcopy(initial),
        RandomizedData(explicit_generator),
        progress_callback=stop,
        cancel_token=token,
    )
    resumed_model = copy.deepcopy(initial)
    resumed = Trainer(cfg).fit(
        resumed_model,
        RandomizedData(explicit_generator),
        resume_dir=stopped.model_path.parent,
    )
    assert resumed.history == continuous.history
    assert resumed.best_epoch == continuous.best_epoch
    assert resumed.best_val_loss == continuous.best_val_loss
    report = json.loads((resumed.model_path.parent / "run.json").read_text("utf-8"))
    assert report["runtime"]["loader_workers"] == {"train": 0, "val": 0}
    for name, expected in continuous_model.state_dict().items():
        assert torch.equal(resumed_model.state_dict()[name], expected), name


def test_checkpoint_contains_completed_epoch_best_metric(tmp_path: Path) -> None:
    """The checkpoint includes the current epoch's best and patience update."""
    token = CancellationToken()

    def stop(event: dict[str, Any]) -> None:
        if event.get("event") == "epoch_end":
            token.cancel()

    result = Trainer(_config(tmp_path)).fit(
        nn.Linear(4, 1), RandomizedData(), progress_callback=stop, cancel_token=token
    )
    state = load_resume_state(result.model_path.parent)
    assert state is not None
    assert state.best_epoch == result.best_epoch == 1
    assert state.best_metric == result.best_val_loss
    assert state.patience_counter == 0


class TaskDataset(RandomizedDataset):
    """Adapt stochastic examples to each standalone task's real tensor contract."""

    def __init__(self, task: str) -> None:
        self.task = task

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        value, label = super().__getitem__(index)
        if self.task == "regression":
            return value, label.float().reshape(1)
        value = value.reshape(1, 2, 2)
        if self.task == "segmentation":
            return value, label.expand(2, 2)
        return value, label


class TaskData:
    """Real shuffled loaders for the standalone training loops."""

    def __init__(self, task: str) -> None:
        self.task = task

    def train_loader(self) -> DataLoader:
        """Build a deterministic sampler with stochastic dataset transforms."""
        return DataLoader(
            TaskDataset(self.task),
            batch_size=4,
            shuffle=True,
            generator=torch.Generator().manual_seed(11),
        )

    def val_loader(self) -> DataLoader:
        """Build an evaluation loader."""
        return DataLoader(TaskDataset(self.task), batch_size=4)

    def test_loader(self) -> DataLoader:
        """Build anomaly evaluation batches."""
        return self.val_loader()

    def calibration_loader(self) -> DataLoader:
        """Build anomaly calibration batches."""
        return self.val_loader()


class TinyAutoencoder(ConvAutoencoder):
    """Minimal reconstruction model implementing the autoencoder score contract."""

    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.layers = nn.Sequential(nn.Dropout(0.4), nn.Conv2d(1, 1, 1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Reconstruct the small input with stochastic training activations."""
        return self.layers(inputs)


@pytest.mark.parametrize("task", ["regression", "segmentation", "anomaly"])
def test_standalone_stochastic_resume_matches_continuous(
    tmp_path: Path, task: str
) -> None:
    """Every native standalone loop restores the same stochastic trajectory."""
    base = _config(tmp_path)
    payload = base.model_dump()
    for key in ("task", "model"):
        payload.pop(key)
    trainer_factory: Callable[
        [], RegressionTrainer | SegmentationTrainer | AnomalyTrainer
    ]
    if task == "regression":
        payload["model"] = {"name": "resnet18", "num_targets": 1, "pretrained": False}
        payload["data"] = {
            "base_dir": str(tmp_path / "dataset"),
            "target_columns": ["target"],
        }
        trainer_factory = partial(
            RegressionTrainer, RegressionConfig.model_validate(payload)
        )
        initial = nn.Sequential(nn.Dropout(0.4), nn.Linear(4, 1))
    elif task == "segmentation":
        payload["model"] = {"name": "unet", "num_classes": 2, "pretrained": False}
        payload["data"] = {"base_dir": str(tmp_path / "dataset")}
        trainer_factory = partial(
            SegmentationTrainer, SegmentationConfig.model_validate(payload)
        )
        initial = nn.Sequential(nn.Dropout(0.4), nn.Conv2d(1, 2, 1))
    else:
        payload["model"] = {"name": "autoencoder", "latent_dim": 8}
        payload["data"] = {"base_dir": str(tmp_path / "dataset"), "image_size": 32}
        trainer_factory = partial(AnomalyTrainer, AnomalyConfig.model_validate(payload))
        initial = TinyAutoencoder()
    continuous_model = copy.deepcopy(initial)
    continuous = trainer_factory().fit(continuous_model, TaskData(task))
    token = CancellationToken()

    def stop(event: dict[str, Any]) -> None:
        if event.get("event") == "epoch_end" and event.get("epoch") == 1:
            token.cancel()

    stopped = trainer_factory().fit(
        copy.deepcopy(initial),
        TaskData(task),
        progress_callback=stop,
        cancel_token=token,
    )
    state = load_resume_state(stopped.model_path.parent)
    assert state is not None and state.best_epoch == stopped.best_epoch
    resumed_model = copy.deepcopy(initial)
    resumed = trainer_factory().fit(
        resumed_model, TaskData(task), resume_dir=stopped.model_path.parent
    )
    assert resumed.history == continuous.history
    assert resumed.best_epoch == continuous.best_epoch
    report = json.loads((resumed.model_path.parent / "run.json").read_text("utf-8"))
    expected_workers = (
        {"train": 0, "test": 0, "calibration": 0}
        if task == "anomaly"
        else {"train": 0, "val": 0}
    )
    assert report["runtime"]["loader_workers"] == expected_workers
    for name, expected in continuous_model.state_dict().items():
        assert torch.equal(resumed_model.state_dict()[name], expected), name


def test_changed_dataset_refuses_resume_without_touching_model_or_checkpoint(
    tmp_path: Path,
) -> None:
    """Dataset identity is checked before any training state can be mutated."""
    cfg = _config(tmp_path)
    sample = tmp_path / "dataset" / "sample.bin"
    sample.write_bytes(b"original")
    token = CancellationToken()

    def stop(event: dict[str, Any]) -> None:
        if event.get("event") == "epoch_end":
            token.cancel()

    stopped = Trainer(cfg).fit(
        nn.Linear(4, 1), RandomizedData(), progress_callback=stop, cancel_token=token
    )
    run = stopped.model_path.parent
    checkpoint = (run / "resume.pt").read_bytes()
    report = (run / "run.json").read_bytes()
    sample.write_bytes(b"changed dataset content")
    model = nn.Linear(4, 1)
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="dataset.*changed"):
        Trainer(cfg).fit(model, RandomizedData(), resume_dir=run)
    for name, value in before.items():
        assert torch.equal(model.state_dict()[name], value)
    assert (run / "resume.pt").read_bytes() == checkpoint
    assert (run / "run.json").read_bytes() == report


class FixedValidationData(RandomizedData):
    """Keep validation constant to isolate the early-stopping counter."""

    def val_loader(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Return identical validation inputs and labels every epoch."""
        return [(torch.ones(4, 4), torch.tensor([0, 1, 0, 1]))]


@pytest.mark.parametrize("stop_epoch", [2, 3])
def test_resumed_patience_stops_on_the_same_epoch(
    tmp_path: Path, stop_epoch: int
) -> None:
    """A completed non-improving epoch counts toward early stopping after resume."""
    cfg = _config(tmp_path)
    cfg.training.epochs = 8
    cfg.training.early_stopping_patience = 2
    initial = nn.Sequential(nn.Dropout(0.4), nn.Linear(4, 1))
    continuous_model = copy.deepcopy(initial)
    continuous = Trainer(cfg).fit(
        continuous_model,
        FixedValidationData(),
        optimizer=torch.optim.SGD(continuous_model.parameters(), lr=0),
    )
    token = CancellationToken()

    def stop(event: dict[str, Any]) -> None:
        if event.get("event") == "epoch_end" and event.get("epoch") == stop_epoch:
            token.cancel()

    stopped_model = copy.deepcopy(initial)
    stopped = Trainer(cfg).fit(
        stopped_model,
        FixedValidationData(),
        optimizer=torch.optim.SGD(stopped_model.parameters(), lr=0),
        progress_callback=stop,
        cancel_token=token,
    )
    state = load_resume_state(stopped.model_path.parent)
    assert state is not None and state.patience_counter == stop_epoch - 1
    resumed_model = copy.deepcopy(initial)
    resumed = Trainer(cfg).fit(
        resumed_model,
        FixedValidationData(),
        optimizer=torch.optim.SGD(resumed_model.parameters(), lr=0),
        resume_dir=stopped.model_path.parent,
    )
    assert continuous.total_epochs == resumed.total_epochs == 3
    assert continuous.history == resumed.history
    assert continuous.best_epoch == resumed.best_epoch == 1


class TinyDetector(nn.Module):
    """Train a real stochastic box-loss head with the torchvision call contract."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Dropout(0.4), nn.Linear(4, 1))

    def forward(self, images: list[torch.Tensor], targets: Any = None) -> Any:
        """Return a differentiable loss or empty detection predictions."""
        if self.training:
            outputs = self.layers(torch.stack(images))
            return {"loss_box_reg": outputs.square().mean()}
        return [
            {
                "boxes": torch.zeros(0, 4),
                "labels": torch.zeros(0, dtype=torch.int64),
                "scores": torch.zeros(0),
            }
            for _ in images
        ]


def _detection_collate(batch: Any) -> Any:
    images, _ = zip(*batch, strict=True)
    return list(images), [
        {"boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]), "labels": torch.tensor([1])}
        for _ in images
    ]


def test_torchvision_stochastic_resume_matches_continuous(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The native detector restores dropout, shuffle and augmentation RNGs."""
    base = _config(tmp_path)
    cfg = DetectionConfig.model_validate(
        {
            "name": "detection_trajectory",
            "model": {
                "backend": "torchvision",
                "name": "fasterrcnn_resnet50_fpn",
                "num_classes": 2,
                "pretrained": False,
            },
            "data": {"base_dir": str(tmp_path / "dataset")},
            "training": {"epochs": 3, "workers": 0, "batch_size": 4, "seed": 19},
            "output": base.output.model_dump(),
            "device": {"kind": "cpu"},
        }
    )
    models: list[TinyDetector] = []

    def build_model(*args: Any) -> TinyDetector:
        model = TinyDetector()
        models.append(model)
        return model

    def build_loaders(
        self: DetectionTrainer, path: Path
    ) -> tuple[DataLoader, DataLoader]:
        return (
            DataLoader(
                RandomizedDataset(),
                batch_size=4,
                shuffle=True,
                generator=torch.Generator().manual_seed(37),
                collate_fn=_detection_collate,
            ),
            DataLoader(
                RandomizedDataset(), batch_size=4, collate_fn=_detection_collate
            ),
        )

    monkeypatch.setattr(
        "visionforge.core.detection_trainer.build_torchvision_detector", build_model
    )
    monkeypatch.setattr(DetectionTrainer, "_build_loaders", build_loaders)
    continuous = DetectionTrainer(cfg).fit()
    token = CancellationToken()

    def stop(event: dict[str, Any]) -> None:
        if event.get("event") == "epoch_end" and event.get("epoch") == 1:
            token.cancel()

    stopped = DetectionTrainer(cfg).fit(progress_callback=stop, cancel_token=token)
    state = load_resume_state(stopped.run_dir)
    assert state is not None and state.best_epoch == stopped.best_epoch == 1
    resumed = DetectionTrainer(cfg).fit(resume_dir=stopped.run_dir)
    assert resumed.history == continuous.history
    report = json.loads((resumed.run_dir / "run.json").read_text("utf-8"))
    assert report["runtime"]["loader_workers"] == {"train": 0, "val": 0}
    assert resumed.best_epoch == continuous.best_epoch
    for name, expected in models[0].state_dict().items():
        assert torch.equal(models[2].state_dict()[name], expected), name


def test_async_best_checkpoint_owns_submission_time_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A delayed best-weight writer must not observe the next optimizer step."""
    cfg = _config(tmp_path)
    cfg.training.epochs = 2
    torch.manual_seed(7)
    model = nn.Sequential(nn.Dropout(0.4), nn.Linear(4, 1))
    writer_started = Event()
    release_writer = Event()
    original_save = torch.save
    original_train = Trainer._train_epoch
    expected: dict[str, torch.Tensor] = {}
    epochs = 0

    def blocked_save(payload: Any, path: Any, *args: Any, **kwargs: Any) -> None:
        if Path(path).name == "best_model.pth":
            writer_started.set()
            assert release_writer.wait(10), "test did not release checkpoint writer"
        original_save(payload, path, *args, **kwargs)

    def train_epoch(self: Trainer, *args: Any, **kwargs: Any) -> Any:
        nonlocal epochs
        if epochs == 1:
            assert writer_started.wait(10), "checkpoint writer did not start"
        result = original_train(self, *args, **kwargs)
        epochs += 1
        if epochs == 1:
            expected.update(copy.deepcopy(model.state_dict()))
        else:
            release_writer.set()
        return result

    monkeypatch.setattr(torch, "save", blocked_save)
    monkeypatch.setattr(Trainer, "_train_epoch", train_epoch)
    try:
        result = Trainer(cfg).fit(model, RandomizedData())
    finally:
        release_writer.set()
    assert result.best_epoch == 1
    saved = torch.load(result.model_path, weights_only=True)
    assert saved._metadata == model.state_dict()._metadata
    for name, weights in expected.items():
        assert torch.equal(saved[name], weights), name
