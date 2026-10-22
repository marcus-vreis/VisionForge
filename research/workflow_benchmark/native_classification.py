"""Independent CPU recipe for engineering checks of paired classification runs."""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    targets: list[int] | None = None,
    predictions: list[int] | None = None,
) -> tuple[float, float]:
    model.train(optimizer is not None)
    total_loss = 0.0
    correct = 0
    count = 0
    with torch.set_grad_enabled(optimizer is not None):
        for images, labels in loader:
            images, labels = images.to("cpu"), labels.to("cpu")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            predicted = logits.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            count += labels.size(0)
            if targets is not None:
                targets.extend(labels.tolist())
            if predictions is not None:
                predictions.extend(predicted.tolist())
    # Trainer averages batch means, including a potentially shorter final batch.
    return total_loss / len(loader), correct / count


def run_native(
    dataset: Path,
    initial_weights: Path,
    output: Path,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    image_size: int = 32,
    learning_rate: float = 0.01,
) -> dict[str, Any]:
    """Run the restricted CPU ResNet18 recipe with shared initial weights."""
    setup_start = time.perf_counter()
    dataset = Path(dataset).resolve()
    initial_weights = Path(initial_weights).resolve()
    output = Path(output).resolve()
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if batch_size < 2 or batch_size & (batch_size - 1):
        raise ValueError("batch_size must be a power of two and at least 2")
    if image_size < 32:
        raise ValueError("image_size must be at least 32")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if output == dataset or dataset in output.parents:
        raise ValueError("output must be outside dataset")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if not initial_weights.is_file():
        raise FileNotFoundError(f"initial weights not found: {initial_weights}")

    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    splits: dict[str, datasets.ImageFolder] = {}
    for split in ("train", "val", "test"):
        root = dataset / split
        if not root.is_dir():
            raise ValueError(f"missing dataset split: {split}")
        splits[split] = datasets.ImageFolder(root, transform=transform)
    class_to_idx = splits["train"].class_to_idx
    if len(class_to_idx) < 2:
        raise ValueError("train must contain at least two classes")
    for split in ("val", "test"):
        if splits[split].class_to_idx != class_to_idx:
            raise ValueError(f"{split} class_to_idx differs from train")
    if image_size == 32 and len(splits["train"]) % batch_size == 1:
        raise ValueError("32px ResNet18 requires a final train batch of at least 2")

    loaders = {
        split: DataLoader(
            images,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=0,
            pin_memory=False,
        )
        for split, images in splits.items()
    }
    with torch.device("cpu"):
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(class_to_idx))
    model.load_state_dict(
        torch.load(initial_weights, map_location="cpu", weights_only=True)
    )
    initial_hash = _sha256(initial_weights)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0)
    criterion = nn.CrossEntropyLoss()
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = output / "best_model.pth"
    history: list[dict[str, int | float]] = []
    best_epoch = 0
    best_val_loss = float("inf")
    setup_seconds = time.perf_counter() - setup_start

    train_start = time.perf_counter()
    random.seed(seed)
    np.random.seed(seed)
    # Seed only the CPU generator; accelerator state is outside this recipe.
    torch.random.default_generator.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = _epoch(model, loaders["train"], criterion, optimizer)
        val_loss, val_acc = _epoch(model, loaders["val"], criterion)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
            }
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            frozen = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            torch.save(frozen, checkpoint)
    train_seconds = time.perf_counter() - train_start

    evaluation_start = time.perf_counter()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    test_targets: list[int] = []
    test_predictions: list[int] = []
    _, test_accuracy = _epoch(
        model,
        loaders["test"],
        criterion,
        targets=test_targets,
        predictions=test_predictions,
    )
    evaluation_seconds = time.perf_counter() - evaluation_start
    result: dict[str, Any] = {
        "workflow": "native_pytorch",
        "device": "cpu",
        "seed": seed,
        "initial_weights_sha256": initial_hash,
        "class_to_idx": class_to_idx,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_accuracy": test_accuracy,
        "test_targets": test_targets,
        "test_predictions": test_predictions,
        "best_model": str(checkpoint),
        "setup_seconds": setup_seconds,
        "train_seconds": train_seconds,
        "evaluation_seconds": evaluation_seconds,
    }
    (output / "native_result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result
