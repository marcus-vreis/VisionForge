import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from visionforge.core.data import DataModule
from visionforge.core.pin_memory import effective_pin_memory, pin_memory_for
from visionforge.utils.config import DeviceConfig, ExperimentConfig

_PIN_WARNING = "no accelerator is found"


def test_requested_on_cpu_is_off() -> None:
    assert effective_pin_memory(True, torch.device("cpu")) is False


def test_requested_on_cuda_is_on() -> None:
    assert effective_pin_memory(True, torch.device("cuda", 0)) is True


def test_not_requested_on_cuda_is_off() -> None:
    assert effective_pin_memory(False, torch.device("cuda", 0)) is False


def test_requested_on_mps_is_off() -> None:
    assert effective_pin_memory(True, torch.device("mps")) is False


def test_cuda_kind_follows_availability() -> None:
    expected = torch.cuda.is_available()
    assert pin_memory_for(True, DeviceConfig(kind="cuda")) is expected


def test_cpu_kind_never_pins() -> None:
    assert pin_memory_for(True, DeviceConfig(kind="cpu")) is False


def test_mps_kind_never_pins() -> None:
    assert pin_memory_for(True, DeviceConfig(kind="mps")) is False


@pytest.fixture
def cpu_config(tmp_path: Path) -> ExperimentConfig:
    for split in ("train", "val", "test"):
        for cls in ("class_a", "class_b"):
            folder = tmp_path / split / cls
            folder.mkdir(parents=True)
            Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(
                folder / "image.png"
            )
    return ExperimentConfig.model_validate(
        {
            "name": "pin",
            "task": "binary",
            "model": {"name": "resnet18", "num_classes": 1, "pretrained": False},
            "training": {"learning_rate": 0.001, "epochs": 1, "batch_size": 2},
            "data": {"base_dir": str(tmp_path), "num_workers": 0, "pin_memory": True},
            "device": {"kind": "cpu"},
        }
    )


def test_cpu_dataloader_emits_no_pin_memory_warning(
    cpu_config: ExperimentConfig,
) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        next(iter(DataModule(cpu_config).train_loader()))
    messages = [str(w.message) for w in caught if _PIN_WARNING in str(w.message)]
    assert messages == []


def test_cpu_dataloader_does_not_pin(cpu_config: ExperimentConfig) -> None:
    assert DataModule(cpu_config).train_loader().pin_memory is False
