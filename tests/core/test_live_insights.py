"""Live Grad-CAM during classification training (ADR-162).

The rule the tests hold it to is that the training does not notice it: same
seed, same history and same weights with or without it.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader, Dataset

from visionforge.core.live_insights import (
    LIVE_GRADCAM_SAMPLES,
    LiveGradCam,
    choose_samples,
    gradcam_batch,
    preserved_rng,
)
from visionforge.core.trainer import Trainer
from visionforge.utils.config import ExperimentConfig


class _TinyCnn(nn.Module):
    """A conv layer for Grad-CAM to hook, dropout and BatchNorm for the mode checks."""

    def __init__(self, outputs: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU()
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.5), nn.Linear(4 * 8 * 8, outputs)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class _Images(Dataset[tuple[torch.Tensor, int]]):
    """Fixed random images with labels, read by index like an ImageFolder."""

    def __init__(self, n: int, classes: int, seed: int) -> None:
        gen = torch.Generator().manual_seed(seed)
        self.images = torch.randn(n, 3, 8, 8, generator=gen)
        self.targets = [i % classes for i in range(n)]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.images[index], self.targets[index]


class _DataModule:
    class_names = ["a", "b", "c"]

    def __init__(self) -> None:
        self._train = _Images(24, 3, seed=1)
        self._val = _Images(12, 3, seed=2)

    def train_loader(self) -> DataLoader[Any]:
        # Shuffled: an insight that moved the global generator would reorder the
        # batches of the next epoch, and the history would differ.
        return DataLoader(self._train, batch_size=8, shuffle=True)

    def val_loader(self) -> DataLoader[Any]:
        return DataLoader(self._val, batch_size=8)


def _config(tmp_path: Path, every: int, name: str) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "name": name,
            "task": "multiclass",
            "model": {"name": "resnet18", "num_classes": 3, "pretrained": False},
            "training": {
                "learning_rate": 0.01,
                "epochs": 3,
                "batch_size": 8,
                "seed": 7,
                "deterministic": True,
                "live_gradcam_every": every,
            },
            "data": {"base_dir": str(tmp_path)},
            "output": {"models_dir": str(tmp_path / "models")},
            "device": {"kind": "cpu"},
        }
    )


def _fit(
    tmp_path: Path, every: int, name: str
) -> tuple[list[dict[str, Any]], dict, list]:
    events: list[dict[str, Any]] = []
    torch.manual_seed(0)
    model = _TinyCnn()
    result = Trainer(_config(tmp_path, every, name)).fit(
        model, _DataModule(), progress_callback=events.append
    )
    run_json = json.loads((result.model_path.parent / "run.json").read_text("utf-8"))
    return (
        events,
        {k: v.clone() for k, v in model.state_dict().items()},
        run_json["history"],
    )


def test_the_training_does_not_notice_the_insights(tmp_path: Path) -> None:
    """Same seed, CPU, deterministic: identical history and weights."""
    _, weights_off, history_off = _fit(tmp_path, 0, "off")
    events_on, weights_on, history_on = _fit(tmp_path, 1, "on")

    assert any(e["event"] == "insight" for e in events_on)
    assert history_on == history_off
    assert weights_on.keys() == weights_off.keys()
    for key, value in weights_off.items():
        assert torch.equal(weights_on[key], value), key


def test_the_insight_event_names_its_images(tmp_path: Path) -> None:
    events, _, _ = _fit(tmp_path, 2, "named")
    insights = [e for e in events if e["event"] == "insight"]
    # Every 2 epochs, plus the last and each new best: at least epochs 2 and 3.
    assert {e["epoch"] for e in insights} >= {2, 3}
    first = insights[0]
    assert first["kind"] == "gradcam"
    assert len(first["images"]) == LIVE_GRADCAM_SAMPLES
    for image in first["images"]:
        assert Path(image["path"]).is_file()
        assert image["true_class"] in _DataModule.class_names
        assert image["predicted_class"] in _DataModule.class_names
    run_dir = Path(first["images"][0]["path"]).parents[2]
    assert (run_dir / "insights" / "best").is_dir()


def test_off_by_default_writes_nothing(tmp_path: Path) -> None:
    events, _, _ = _fit(tmp_path, 0, "default_off")
    assert not any(e["event"] == "insight" for e in events)
    assert not list((tmp_path / "models").rglob("insights"))


# ── units ─────────────────────────────────────────────────────────────────────


def test_choose_samples_takes_the_classes_in_turn() -> None:
    labels = [0] * 10 + [1] * 2 + [2] * 5
    chosen = choose_samples(labels, 6, seed=3)
    assert len(chosen) == len(set(chosen)) == 6
    # Class 1 has two images and both show, though it is the rarest.
    assert sorted(labels[i] for i in chosen) == [0, 0, 1, 1, 2, 2]
    assert choose_samples(labels, 6, seed=3) == chosen


def test_choose_samples_with_fewer_images_than_asked() -> None:
    assert sorted(choose_samples([0, 1, 1], 6, seed=0)) == [0, 1, 2]


def test_preserved_rng_leaves_every_generator_as_found() -> None:
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    expected = (random.random(), np.random.rand(), torch.rand(1).item())

    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    with preserved_rng(torch.device("cpu")):
        random.random()
        np.random.rand(5)
        torch.rand(5)
        torch.randperm(10)
    after = (random.random(), np.random.rand(), torch.rand(1).item())
    assert after == expected


def test_gradcam_batch_leaves_parameter_grads_alone() -> None:
    torch.manual_seed(0)
    model = _TinyCnn().eval()
    images = torch.randn(4, 3, 8, 8)
    cams, predicted = gradcam_batch(model, model.features[0], images)

    assert cams.shape == (4, 8, 8)
    assert cams.dtype == torch.float32
    assert torch.isfinite(cams).all()
    assert cams.min() >= 0.0
    assert cams.max() <= 1.0
    assert torch.equal(predicted, model(images).argmax(dim=1))
    assert all(p.grad is None for p in model.parameters())


def test_gradcam_batch_scores_the_predicted_side_of_a_single_logit() -> None:
    torch.manual_seed(0)
    model = _TinyCnn(outputs=1).eval()
    images = torch.randn(3, 3, 8, 8)
    _, predicted = gradcam_batch(model, model.features[0], images)
    assert torch.equal(predicted, (model(images)[:, 0] > 0).long())


def test_gradcam_batch_with_a_frozen_backbone() -> None:
    """Feature extraction: the input's grad keeps the activation in the graph."""
    model = _TinyCnn().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    cams, _ = gradcam_batch(model, model.features[0], torch.randn(2, 3, 8, 8))
    assert torch.isfinite(cams).all()


def _insights(tmp_path: Path, every: int = 1) -> LiveGradCam:
    cfg = _config(tmp_path, every, "unit")
    loader = DataLoader(_Images(9, 3, seed=4), batch_size=4)
    insights = LiveGradCam.create(
        cfg, loader, ["a", "b", "c"], tmp_path, torch.device("cpu")
    )
    assert insights is not None
    return insights


def test_after_epoch_restores_training_mode(tmp_path: Path) -> None:
    model = _TinyCnn().train()
    running_mean = model.features[1].running_mean.clone()
    _insights(tmp_path).after_epoch(model, 1, 0.5, None)
    assert model.training
    assert torch.equal(model.features[1].running_mean, running_mean)


def test_schedule_every_n_the_last_and_each_best(tmp_path: Path) -> None:
    insights = _insights(tmp_path, every=2)
    assert insights.due(2, is_best=False)
    assert not insights.due(1, is_best=False)
    assert insights.due(1, is_best=True)
    assert insights.due(3, is_best=False)  # the last of three epochs


def test_a_failure_is_a_warning_and_ends_the_insights(tmp_path: Path) -> None:
    class NoConv(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.flatten(1)[:, :3]

    warnings: list[str] = []
    handle = logger.add(lambda m: warnings.append(str(m)), level="WARNING")
    events: list[dict[str, Any]] = []
    insights = _insights(tmp_path)
    try:
        insights.after_epoch(NoConv(), 1, 0.5, events.append)
        insights.after_epoch(NoConv(), 2, 0.4, events.append)
    finally:
        logger.remove(handle)
    assert events == []
    assert len([w for w in warnings if "Live Grad-CAM failed" in w]) == 1


def test_off_without_labels_to_choose_from(tmp_path: Path) -> None:
    cfg = _config(tmp_path, 1, "nolabels")
    batches = [(torch.randn(2, 3, 8, 8), torch.tensor([0, 1]))]
    assert (
        LiveGradCam.create(cfg, batches, ["a"], tmp_path, torch.device("cpu")) is None
    )


@pytest.mark.parametrize("every", [0])
def test_create_is_none_when_off(tmp_path: Path, every: int) -> None:
    cfg = _config(tmp_path, every, "off_unit")
    loader = DataLoader(_Images(3, 3, seed=0))
    assert LiveGradCam.create(cfg, loader, ["a"], tmp_path, torch.device("cpu")) is None
