from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, models, transforms

from research.workflow_benchmark.native_classification import _epoch, run_native


@pytest.fixture
def recipe(tmp_path: Path) -> tuple[Path, Path]:
    """Create real ImageFolder splits and a shared ResNet18 state dictionary."""
    dataset = tmp_path / "dataset"
    for split_index, split in enumerate(("train", "val", "test")):
        for class_index, label in enumerate(("amber", "blue")):
            folder = dataset / split / label
            folder.mkdir(parents=True)
            for index in range(3):
                pixels = np.random.default_rng(
                    100 * split_index + 10 * class_index + index
                ).integers(0, 256, (36, 40, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(folder / f"{index}.png")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    weights = tmp_path / "initial.pth"
    torch.save(model.state_dict(), weights)
    return dataset, weights


def test_native_repeats_exactly_and_evaluates_best_checkpoint(
    recipe: tuple[Path, Path], tmp_path: Path
) -> None:
    """Check the real CPU path, saved evidence and deterministic repetition."""
    dataset, weights = recipe
    results = [
        run_native(dataset, weights, tmp_path / name, seed=37, epochs=2, batch_size=4)
        for name in ("first", "repeat")
    ]
    first, repeat = results
    ignored = {"phases", "best_model"}
    assert {k: v for k, v in first.items() if k not in ignored} == {
        k: v for k, v in repeat.items() if k not in ignored
    }
    assert first["workflow"] == "native_pytorch"
    assert first["device"] == "cpu"
    assert first["class_to_idx"] == {"amber": 0, "blue": 1}
    assert first["test_targets"] == [0, 0, 0, 1, 1, 1]
    assert (
        first["test_accuracy"]
        == sum(
            target == predicted
            for target, predicted in zip(
                first["test_targets"], first["test_predictions"], strict=True
            )
        )
        / 6
    )
    assert (
        first["initial_weights_sha256"]
        == hashlib.sha256(weights.read_bytes()).hexdigest()
    )
    assert [row["epoch"] for row in first["history"]] == [1, 2]
    best = min(first["history"], key=lambda row: row["val_loss"])
    assert first["best_epoch"] == best["epoch"]
    assert first["best_val_loss"] == best["val_loss"]
    for result in results:
        checkpoint = Path(result["best_model"])
        assert checkpoint.is_file()
        assert (
            json.loads((checkpoint.parent / "native_result.json").read_text()) == result
        )
        assert [phase["seconds"] >= 0 for phase in result["phases"].values()] == [
            True
        ] * 4
        for row in result["history"]:
            assert set(row) == {
                "epoch",
                "train_loss",
                "val_loss",
                "train_acc",
                "val_acc",
            }
            assert 0 <= row["train_acc"] <= 1
            assert 0 <= row["val_acc"] <= 1
    first_state = torch.load(first["best_model"], map_location="cpu", weights_only=True)
    repeated_state = torch.load(
        repeat["best_model"], map_location="cpu", weights_only=True
    )
    for name, tensor in first_state.items():
        assert tensor.device.type == "cpu"
        assert torch.equal(tensor, repeated_state[name]), name
    initial_state = torch.load(weights, map_location="cpu", weights_only=True)
    assert not torch.equal(first_state["conv1.weight"], initial_state["conv1.weight"])

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(first_state)
    transform = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    test_loader = DataLoader(
        datasets.ImageFolder(dataset / "test", transform=transform), batch_size=4
    )
    val_loader = DataLoader(
        datasets.ImageFolder(dataset / "val", transform=transform), batch_size=4
    )
    predictions: list[int] = []
    assert (
        _epoch(model, test_loader, nn.CrossEntropyLoss(), predictions=predictions)[1]
        == first["test_accuracy"]
    )
    assert predictions == first["test_predictions"]
    assert _epoch(model, val_loader, nn.CrossEntropyLoss())[0] == first["best_val_loss"]


def test_epoch_averages_batch_losses_and_weights_accuracy_by_sample() -> None:
    """Preserve the Trainer recipe on an uneven final batch."""
    logits = torch.tensor([[8.0, 0.0]] * 4 + [[0.0, 8.0]])
    labels = torch.zeros(5, dtype=torch.long)
    loader = DataLoader(TensorDataset(logits, labels), batch_size=4)
    criterion = nn.CrossEntropyLoss()
    loss, accuracy = _epoch(nn.Identity(), loader, criterion)
    expected = (
        criterion(logits[:4], labels[:4]) + criterion(logits[4:], labels[4:])
    ) / 2
    assert loss == pytest.approx(expected.item())
    assert loss != pytest.approx(criterion(logits, labels).item())
    assert accuracy == 4 / 5


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"epochs": 0}, "epochs"),
        ({"batch_size": 1}, "batch_size"),
        ({"batch_size": 3}, "batch_size"),
        ({"image_size": 31}, "image_size"),
        ({"learning_rate": 0}, "learning_rate"),
        ({"learning_rate": float("nan")}, "learning_rate"),
        ({"learning_rate": float("inf")}, "learning_rate"),
    ],
)
def test_invalid_parameters_leave_no_output(
    tmp_path: Path, overrides: dict[str, Any], match: str
) -> None:
    """Reject invalid controls before reading data or creating artifacts."""
    output = tmp_path / "output"
    kwargs = {"seed": 37, "epochs": 1, "batch_size": 4, **overrides}
    with pytest.raises(ValueError, match=match):
        run_native(tmp_path / "dataset", tmp_path / "initial.pth", output, **kwargs)
    assert not output.exists()


def test_existing_and_unsafe_output_are_rejected(tmp_path: Path) -> None:
    """Never overwrite existing output or write inside the dataset."""
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    existing = tmp_path / "existing"
    existing.mkdir()
    sentinel = existing / "keep.txt"
    sentinel.write_text("untouched", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        run_native(
            dataset, tmp_path / "weights", existing, seed=0, epochs=1, batch_size=4
        )
    assert sentinel.read_text() == "untouched"
    for unsafe in (dataset, dataset / "nested" / "output"):
        with pytest.raises(ValueError, match="outside dataset"):
            run_native(
                dataset, tmp_path / "weights", unsafe, seed=0, epochs=1, batch_size=4
            )
    assert list(dataset.iterdir()) == []


@pytest.mark.parametrize(
    "invalid", ["missing_split", "unknown_class", "missing_class", "one_class"]
)
def test_invalid_splits_leave_no_output(
    recipe: tuple[Path, Path], tmp_path: Path, invalid: str
) -> None:
    """Require all three splits and the same populated class map."""
    dataset, weights = recipe
    if invalid == "missing_split":
        (dataset / "test").rename(dataset / "not_test")
        match = "missing dataset split"
    elif invalid == "unknown_class":
        (dataset / "val" / "blue").rename(dataset / "val" / "unknown")
        match = "class_to_idx differs"
    elif invalid == "missing_class":
        (dataset / "test" / "blue").rename(tmp_path / "held_out")
        match = "class_to_idx differs"
    else:
        (dataset / "train" / "blue").rename(tmp_path / "held_out")
        match = "at least two classes"
    output = tmp_path / "output"
    with pytest.raises(ValueError, match=match):
        run_native(dataset, weights, output, seed=0, epochs=1, batch_size=4)
    assert not output.exists()


def test_missing_weights_leave_no_output(tmp_path: Path) -> None:
    """Require an existing initial state before creating output."""
    output = tmp_path / "output"
    with pytest.raises(FileNotFoundError, match="initial weights"):
        run_native(
            tmp_path / "dataset",
            tmp_path / "weights",
            output,
            seed=0,
            epochs=1,
            batch_size=4,
        )
    assert not output.exists()


def test_single_item_final_training_batch_is_rejected(
    recipe: tuple[Path, Path], tmp_path: Path
) -> None:
    """Reject a 32px singleton train batch before BatchNorm or output creation."""
    dataset, weights = recipe
    (dataset / "train" / "blue" / "2.png").unlink()
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="final train batch"):
        run_native(dataset, weights, output, seed=0, epochs=1, batch_size=4)
    assert not output.exists()
