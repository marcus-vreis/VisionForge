"""`training.deterministic` must mean the same thing in every task (ADR-062).

One file on purpose: the defect this guards against is a *gap* — a task that
never got the knob, or got the field but never forwards it — and a per-task
test file is exactly how such a gap stays invisible. Parametrizing over the
config classes makes a new task that forgets the field fail here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
from pydantic import BaseModel

from visionforge.core.anomaly_trainer import AnomalyTrainer
from visionforge.core.regression_trainer import RegressionTrainer
from visionforge.core.segmentation_trainer import SegmentationTrainer
from visionforge.models.anomaly_factory import ConvAutoencoder
from visionforge.tasks.base import TaskTrainingConfig
from visionforge.utils.anomaly_config import AnomalyConfig, AnomalyTrainingConfig
from visionforge.utils.config import TrainingConfig
from visionforge.utils.detection_config import DetectionConfig, DetectionTrainingConfig
from visionforge.utils.regression_config import (
    RegressionConfig,
    RegressionTrainingConfig,
)
from visionforge.utils.segmentation_config import (
    SegmentationConfig,
    SegmentationTrainingConfig,
)

# Every training config in the project, with the default each one documents.
# Detection is True because its stated contract is that an unmodified config
# trains exactly like a bare `YOLO.train` call, and Ultralytics defaults to True.
TRAINING_CONFIGS = [
    pytest.param(TrainingConfig, False, id="classification"),
    pytest.param(RegressionTrainingConfig, False, id="regression"),
    pytest.param(SegmentationTrainingConfig, False, id="segmentation"),
    pytest.param(AnomalyTrainingConfig, False, id="anomaly"),
    pytest.param(DetectionTrainingConfig, True, id="detection"),
    pytest.param(TaskTrainingConfig, False, id="custom-sdk"),
]


class TestConfigParity:
    @pytest.mark.parametrize(("config_cls", "default"), TRAINING_CONFIGS)
    def test_every_task_has_the_knob_with_its_documented_default(
        self, config_cls: type[BaseModel], default: bool
    ) -> None:
        field = config_cls.model_fields.get("deterministic")
        assert field is not None, f"{config_cls.__name__} is missing deterministic"
        assert field.default is default

    @pytest.mark.parametrize(("config_cls", "_default"), TRAINING_CONFIGS)
    def test_the_knob_is_settable_both_ways(
        self, config_cls: type[BaseModel], _default: bool
    ) -> None:
        for value in (True, False):
            parsed = config_cls.model_validate({"deterministic": value})
            # Through model_dump because that is the form that reaches the
            # exported YAML and run.json — a field that parses but does not
            # serialize would still break the re-run contract.
            assert parsed.model_dump()["deterministic"] is value

    @pytest.mark.parametrize(("config_cls", "_default"), TRAINING_CONFIGS)
    def test_the_description_is_shared_not_retyped(
        self, config_cls: type[BaseModel], _default: bool
    ) -> None:
        """A GUI reads these descriptions; four hand-copies would drift."""
        description = config_cls.model_fields["deterministic"].description
        assert description is not None
        assert "bit-exact reproducibility" in description


# ── the knob actually reaches the seeder ──────────────────────────────────────


class _SeedSpy:
    """Records the kwargs `_seed_everything` was called with."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, bool]] = []

    def __call__(self, seed: int, *, deterministic: bool = False) -> None:
        self.calls.append((seed, deterministic))


class _TinySegModel(nn.Module):
    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _TinyRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3 * 32 * 32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(1))


class _FakeLoaders:
    """Two fixed batches, shaped per task."""

    def __init__(self, kind: str) -> None:
        self._kind = kind

    def _batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._kind == "regression":
            return torch.randn(2, 3, 32, 32), torch.randn(2, 1)
        if self._kind == "segmentation":
            return torch.randn(2, 3, 16, 16), torch.randint(0, 3, (2, 16, 16))
        return torch.randn(2, 3, 16, 16), torch.zeros(2, dtype=torch.long)

    def train_loader(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [self._batch() for _ in range(2)]

    def val_loader(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [self._batch() for _ in range(2)]

    def test_loader(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [
            (torch.randn(2, 3, 16, 16), torch.tensor([0, 1])),
            (torch.randn(2, 3, 16, 16), torch.tensor([0, 1])),
        ]


def _base(tmp_path: Path, **training: Any) -> dict[str, Any]:
    return {
        "training": {"epochs": 1, "batch_size": 2, "seed": 7, **training},
        "output": {"models_dir": str(tmp_path / "models")},
        "device": {"kind": "cpu"},
    }


class TestReachesTheSeeder:
    """`_seed_everything(seed, deterministic=...)` — the field is inert without it."""

    @pytest.mark.parametrize("deterministic", [True, False])
    def test_regression(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, deterministic: bool
    ) -> None:
        spy = _SeedSpy()
        monkeypatch.setattr("visionforge.core.regression_trainer._seed_everything", spy)
        cfg = RegressionConfig.model_validate(
            {
                "name": "reg",
                "model": {"name": "resnet18", "num_targets": 1, "pretrained": False},
                "data": {"base_dir": str(tmp_path), "target_columns": ["t"]},
                **_base(tmp_path, deterministic=deterministic),
            }
        )
        RegressionTrainer(cfg).fit(_TinyRegressor(), _FakeLoaders("regression"))
        assert spy.calls == [(7, deterministic)]

    @pytest.mark.parametrize("deterministic", [True, False])
    def test_segmentation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, deterministic: bool
    ) -> None:
        spy = _SeedSpy()
        monkeypatch.setattr(
            "visionforge.core.segmentation_trainer._seed_everything", spy
        )
        cfg = SegmentationConfig.model_validate(
            {
                "name": "seg",
                "model": {"name": "unet", "num_classes": 3, "pretrained": False},
                "data": {"base_dir": str(tmp_path), "image_size": 32},
                **_base(tmp_path, deterministic=deterministic),
            }
        )
        SegmentationTrainer(cfg).fit(_TinySegModel(3), _FakeLoaders("segmentation"))
        assert spy.calls == [(7, deterministic)]

    @pytest.mark.parametrize("deterministic", [True, False])
    def test_anomaly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, deterministic: bool
    ) -> None:
        spy = _SeedSpy()
        monkeypatch.setattr("visionforge.core.anomaly_trainer._seed_everything", spy)
        cfg = AnomalyConfig.model_validate(
            {
                "name": "anom",
                "model": {"name": "autoencoder", "latent_dim": 8},
                "data": {"base_dir": str(tmp_path), "image_size": 32},
                **_base(tmp_path, deterministic=deterministic),
            }
        )
        AnomalyTrainer(cfg).fit(ConvAutoencoder(latent_dim=8), _FakeLoaders("anomaly"))
        assert spy.calls == [(7, deterministic)]


class TestDetectionBothBackends:
    """Detection has two engines; the knob has to reach both."""

    def _config(self, tmp_path: Path, **training: Any) -> DetectionConfig:
        (tmp_path / "images").mkdir(exist_ok=True)
        return DetectionConfig.model_validate(
            {
                "name": "det",
                "model": {"backend": "ultralytics", "name": "yolo11n"},
                "data": {"base_dir": str(tmp_path)},
                **_base(tmp_path, **training),
            }
        )

    @pytest.mark.parametrize("deterministic", [True, False])
    def test_forwarded_to_ultralytics_train(
        self, tmp_path: Path, deterministic: bool
    ) -> None:
        from visionforge.core.detection_trainer import DetectionTrainer

        cfg = self._config(tmp_path, deterministic=deterministic)
        kwargs = DetectionTrainer(cfg)._ultralytics_train_kwargs()
        assert kwargs["deterministic"] is deterministic

    def test_torchvision_backend_is_seeded_at_all(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """It used to forward `seed` only to Ultralytics, so `seed: 42` was a
        claim nothing backed on the torchvision path."""
        from visionforge.core import detection_trainer as module

        spy = _SeedSpy()
        monkeypatch.setattr(module, "_seed_everything", spy)
        cfg = self._config(tmp_path, deterministic=True)
        cfg.model.backend = "torchvision"
        cfg.model.name = "fasterrcnn_resnet50_fpn"

        # The loaders need a real YOLO layout; stopping right after the seed
        # call is enough — this asserts the seeding happens, not that it trains.
        monkeypatch.setattr(
            module,
            "build_torchvision_detector",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stop")),
        )
        with pytest.raises(RuntimeError, match="stop"):
            module.DetectionTrainer(cfg).fit()
        assert spy.calls == [(7, True)]
