"""Independent device-selectable recipe for engineering checks of paired classification runs."""

from __future__ import annotations

if __name__ == "__main__":
    from .paired_classification import main

    main(native=True)
    raise SystemExit

import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from . import phase_contract

NATIVE_INCLUDES = {
    "setup": [
        "ImageFolder indexing of train/val/test",
        "DataLoader construction",
        "ResNet18 construction and initial-weights load",
        "initial-weights SHA-256",
        "SGD optimizer and loss construction",
    ],
    "training": [
        "seeding and deterministic-algorithm switch",
        "per-epoch training and validation passes",
        "best-checkpoint writes",
    ],
    "evaluation": ["best-checkpoint reload", "test pass with loss and predictions"],
    "evidence": ["best-checkpoint SHA-256", "result assembly"],
}


def _epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    device: str = "cpu",
    targets: list[int] | None = None,
    predictions: list[int] | None = None,
) -> tuple[float, float]:
    model.train(optimizer is not None)
    total_loss = 0.0
    correct = 0
    count = 0
    with torch.set_grad_enabled(optimizer is not None):
        for images, labels in loader:
            images, labels = (
                images.to("cuda:0" if device == "cuda" else device),
                labels.to("cuda:0" if device == "cuda" else device),
            )
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


def preflight(
    dataset: Path,
    initial_weights: Path,
    output: Path,
    *,
    epochs: int,
    batch_size: int,
    image_size: int,
    learning_rate: float,
) -> dict[str, int]:
    """Validate controls and split layout without creating output; return class_to_idx."""
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
    class_maps: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        root = dataset / split
        if not root.is_dir():
            raise ValueError(f"missing dataset split: {split}")
        try:
            class_maps[split] = datasets.folder.find_classes(str(root))[1]
        except FileNotFoundError:
            class_maps[split] = {}
    class_to_idx = class_maps["train"]
    if len(class_to_idx) < 2:
        raise ValueError("train must contain at least two classes")
    for split in ("val", "test"):
        if class_maps[split] != class_to_idx:
            raise ValueError(f"{split} class_to_idx differs from train")
    train_count = len(
        datasets.folder.make_dataset(
            str(dataset / "train"), class_to_idx, datasets.folder.IMG_EXTENSIONS
        )
    )
    if image_size == 32 and train_count % batch_size == 1:
        raise ValueError("32px ResNet18 requires a final train batch of at least 2")
    return class_to_idx


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
    concurrent_workload: str = "undeclared",
    device: str = "cpu",
    parity_atol: float | None = None,
) -> dict[str, Any]:
    """Run the restricted ResNet18 recipe under the shared phase contract."""
    parity_atol = phase_contract.validate_device(device, parity_atol)
    dataset = Path(dataset).resolve()
    initial_weights = Path(initial_weights).resolve()
    output = Path(output).resolve()
    class_to_idx = preflight(
        dataset,
        initial_weights,
        output,
        epochs=epochs,
        batch_size=batch_size,
        image_size=image_size,
        learning_rate=learning_rate,
    )
    output.mkdir(parents=True, exist_ok=False)
    recorder = phase_contract.PhaseRecorder(
        "native_pytorch",
        output / "native_result.json",
        NATIVE_INCLUDES,
        phase_contract.capture_conditions(concurrent_workload, device),
    )
    checkpoint = output / "best_model.pth"

    with recorder.phase("setup"):
        transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )
        loaders = {
            split: DataLoader(
                datasets.ImageFolder(dataset / split, transform=transform),
                batch_size=batch_size,
                shuffle=split == "train",
                num_workers=0,
                pin_memory=False,
            )
            for split in ("train", "val", "test")
        }
        with torch.device("cpu"):
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, len(class_to_idx))
        model.load_state_dict(
            torch.load(initial_weights, map_location="cpu", weights_only=True)
        )
        model.to("cuda:0" if device == "cuda" else device)
        initial_hash = phase_contract.sha256_file(initial_weights)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate, weight_decay=0
        )
        criterion = nn.CrossEntropyLoss()

    history: list[dict[str, int | float]] = []
    best_epoch = 0
    best_val_loss = float("inf")
    with recorder.phase("training"):
        random.seed(seed)
        np.random.seed(seed)
        # Keep the CPU seed path unchanged; seed CUDA only when selected.
        torch.random.default_generator.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = _epoch(
                model, loaders["train"], criterion, optimizer, device=device
            )
            val_loss, val_acc = _epoch(model, loaders["val"], criterion, device=device)
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

    test_targets: list[int] = []
    test_predictions: list[int] = []
    with recorder.phase("evaluation"):
        model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True)
        )
        _, test_accuracy = _epoch(
            model,
            loaders["test"],
            criterion,
            device=device,
            targets=test_targets,
            predictions=test_predictions,
        )

    with recorder.phase("evidence"):
        results: dict[str, Any] = {
            "device": device,
            "parity_atol": parity_atol,
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
            "best_model_sha256": phase_contract.sha256_file(checkpoint),
        }
    return recorder.complete(results)
