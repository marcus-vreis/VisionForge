"""Independent CPU recipe for engineering checks of paired segmentation runs (ADR-149).

Mirrors the restricted recipe of ``SegmentationDataModule``/``SegmentationTrainer``
with plain PIL, NumPy and torch, without importing VisionForge.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from . import phase_contract

NUM_CLASSES = 2
IGNORE_INDEX = 255
ALLOWED_MASK_VALUES = frozenset({*range(NUM_CLASSES), IGNORE_INDEX})
NORMALIZE_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_STD = (0.5, 0.5, 0.5)
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
MASK_SUFFIX = ".png"
SPLITS = ("train", "val", "test")
ARCHITECTURE = "tiny_fcn"

NATIVE_INCLUDES = {
    "setup": [
        "image/mask pairing of train/val/test",
        "DataLoader construction",
        "tiny_fcn construction and initial-weights load",
        "initial-weights SHA-256",
        "SGD optimizer and cross-entropy construction",
    ],
    "training": [
        "seeding",
        "per-epoch training and validation passes with confusion metrics",
        "best-checkpoint writes",
    ],
    "evaluation": [
        "best-checkpoint reload",
        "test pass with predictions and per-image confusion",
    ],
    "evidence": [
        "test-prediction file write and SHA-256",
        "best-checkpoint SHA-256",
        "result assembly",
    ],
}


def build_tiny_fcn(num_classes: int) -> nn.Module:
    """Return the harness network: two 3x3 conv+ReLU layers and a 1x1 class head."""
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, num_classes, 1),
    )


def load_pair(
    image_path: Path, mask_path: Path, image_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a normalized float32 [3,S,S] image and a long [S,S] class-id mask.

    The image is resized bilinearly and the mask by nearest neighbour, so mask
    values stay class ids or the ignore value.
    """
    size = (image_size, image_size)
    with Image.open(image_path) as source:
        image = source.convert("RGB").resize(size, Image.Resampling.BILINEAR)
    pixels = torch.from_numpy(np.array(image, dtype=np.uint8)).permute(2, 0, 1)
    mean = torch.tensor(NORMALIZE_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(NORMALIZE_STD, dtype=torch.float32).view(3, 1, 1)
    tensor = (pixels.to(torch.float32).div(255) - mean) / std
    return tensor, torch.from_numpy(_resized_mask(mask_path, image_size)).long()


def _resized_mask(mask_path: Path, image_size: int) -> np.ndarray:
    with Image.open(mask_path) as mask:
        if mask.mode not in ("L", "P"):
            raise ValueError(
                f"mask {mask_path.name} is {mask.mode}; expected single-channel L or P"
            )
        values = np.array(mask)
        resized = mask.resize((image_size, image_size), Image.Resampling.NEAREST)
    unknown = sorted(set(np.unique(values).tolist()) - ALLOWED_MASK_VALUES)
    if unknown:
        raise ValueError(
            f"mask values {unknown} in {mask_path.name}; "
            f"expected {sorted(ALLOWED_MASK_VALUES)}"
        )
    return np.array(resized)


class _PairDataset(Dataset):  # type: ignore[type-arg]
    def __init__(self, pairs: list[tuple[Path, Path]], image_size: int) -> None:
        self._pairs = pairs
        self._size = image_size

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return load_pair(*self._pairs[index], self._size)


def pair_split(split_dir: Path) -> list[tuple[Path, Path]]:
    """Pair ``images/`` with ``masks/`` by stem, sorted by image name.

    Raises:
        ValueError: unsupported, duplicate-stem or unpaired files.
    """
    images, masks = split_dir / "images", split_dir / "masks"
    if not images.is_dir() or not masks.is_dir():
        raise ValueError(f"{split_dir} needs images/ and masks/ directories")
    image_files = sorted(p for p in images.iterdir() if p.is_file())
    mask_files = sorted(p for p in masks.iterdir() if p.is_file())
    unsupported = [
        p.name for p in image_files if p.suffix.lower() not in IMAGE_SUFFIXES
    ] + [p.name for p in mask_files if p.suffix.lower() != MASK_SUFFIX]
    if unsupported:
        raise ValueError(f"unsupported files in {split_dir}: {unsupported}")
    image_stems = [p.stem for p in image_files]
    if len(set(image_stems)) != len(image_stems):
        raise ValueError(f"duplicate image stems in {images}")
    by_stem = {p.stem: p for p in mask_files}
    unpaired = sorted(set(image_stems) ^ set(by_stem))
    if unpaired or not image_files:
        raise ValueError(f"unpaired or missing images/masks in {split_dir}: {unpaired}")
    return [(path, by_stem[path.stem]) for path in image_files]


def preflight(
    dataset: Path,
    initial_weights: Path,
    output: Path,
    *,
    epochs: int,
    batch_size: int,
    image_size: int,
    learning_rate: float,
) -> dict[str, list[tuple[Path, Path]]]:
    """Validate controls, pairs and masks without creating output; return the pairs."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if image_size < 32:
        raise ValueError("image_size must be at least 32")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if output == dataset or dataset in output.parents or output in dataset.parents:
        raise ValueError("output and dataset must be separate directories")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if not initial_weights.is_file():
        raise FileNotFoundError(f"initial weights not found: {initial_weights}")
    pairs = {split: pair_split(dataset / split) for split in SPLITS}
    for split_pairs in pairs.values():
        for _, mask_path in split_pairs:
            # An all-ignored batch makes mean cross-entropy NaN (ADR-149).
            if (_resized_mask(mask_path, image_size) == IGNORE_INDEX).all():
                raise ValueError(
                    f"mask {mask_path.name} is all ignored at {image_size}px"
                )
    return pairs


def confusion_matrix(
    predictions: np.ndarray, targets: np.ndarray, num_classes: int, ignore_index: int
) -> np.ndarray:
    """Count (target row, prediction column) over pixels whose target is not ignored."""
    truth = np.asarray(targets).reshape(-1).astype(np.int64)
    guess = np.asarray(predictions).reshape(-1).astype(np.int64)
    keep = truth != ignore_index
    flat = truth[keep] * num_classes + guess[keep]
    counts = np.bincount(flat, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes)


def segmentation_metrics(confusion: np.ndarray) -> dict[str, float]:
    """Mean IoU and Dice over classes present in truth or prediction, and pixel accuracy.

    Raises:
        ValueError: the matrix holds no labelled pixels.
    """
    matrix = np.asarray(confusion, dtype=np.float64)
    total = matrix.sum()
    if total == 0:
        raise ValueError("confusion matrix has no labelled pixels")
    diag = np.diag(matrix)
    truth, guess = matrix.sum(axis=1), matrix.sum(axis=0)
    present = (truth + guess) > 0
    iou = diag[present] / (truth + guess - diag)[present]
    dice = 2 * diag[present] / (truth + guess)[present]
    return {
        "miou": float(iou.mean()),
        "dice": float(dice.mean()),
        "pixel_acc": float(diag.sum() / total),
    }


class BestCheckpoint:
    """Keep the first epoch with the highest validation mean IoU on disk."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.epoch = 0
        self.miou = -1.0

    def offer(self, epoch: int, miou: float, model: nn.Module) -> bool:
        """Save the model when ``miou`` strictly beats the best so far."""
        if miou <= self.miou:
            return False
        self.epoch, self.miou = epoch, miou
        frozen = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        torch.save(frozen, self.path)
        return True


def _pass(
    model: nn.Module,
    loader: DataLoader,  # type: ignore[type-arg]
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, np.ndarray]:
    """Return the mean batch loss and the split confusion matrix."""
    model.train(optimizer is not None)
    total = 0.0
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    with torch.set_grad_enabled(optimizer is not None):
        for images, masks in loader:
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, masks)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total += loss.item()
            predicted = logits.detach().argmax(dim=1).numpy()
            confusion += confusion_matrix(
                predicted, masks.numpy(), NUM_CLASSES, IGNORE_INDEX
            )
    return total / len(loader), confusion


def _predict(
    model: nn.Module,
    loader: DataLoader,  # type: ignore[type-arg]
) -> tuple[np.ndarray, list[list[list[int]]]]:
    """Return uint8 [N,S,S] predictions and one confusion matrix per image."""
    model.eval()
    predictions: list[np.ndarray] = []
    per_image: list[list[list[int]]] = []
    with torch.inference_mode():
        for images, masks in loader:
            predicted = model(images).argmax(dim=1).numpy()
            for guess, truth in zip(predicted, masks.numpy(), strict=True):
                per_image.append(
                    confusion_matrix(guess, truth, NUM_CLASSES, IGNORE_INDEX).tolist()
                )
            predictions.append(predicted.astype(np.uint8))
    return np.concatenate(predictions), per_image


def save_predictions(path: Path, predictions: np.ndarray) -> None:
    """Write predictions as a new ``.npy`` file; an existing file is never replaced."""
    with Path(path).open("xb") as stream:
        np.save(stream, predictions, allow_pickle=False)


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
) -> dict[str, Any]:
    """Run the restricted CPU segmentation recipe under the shared phase contract."""
    dataset = Path(dataset).resolve()
    initial_weights = Path(initial_weights).resolve()
    output = Path(output).resolve()
    pairs = preflight(
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
        phase_contract.capture_conditions(concurrent_workload),
    )
    best = BestCheckpoint(output / "best_model.pth")

    with recorder.phase("setup"):
        loaders = {
            split: DataLoader(
                _PairDataset(pairs[split], image_size),
                batch_size=batch_size,
                shuffle=split == "train",
                num_workers=0,
                pin_memory=False,
            )
            for split in SPLITS
        }
        model = build_tiny_fcn(NUM_CLASSES)
        model.load_state_dict(
            torch.load(initial_weights, map_location="cpu", weights_only=True)
        )
        initial_hash = phase_contract.sha256_file(initial_weights)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate, weight_decay=0
        )
        criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    history: list[dict[str, float | int]] = []
    with recorder.phase("training"):
        random.seed(seed)
        np.random.seed(seed)
        # CPU generator only; accelerator state is outside this recipe.
        torch.random.default_generator.manual_seed(seed)
        for epoch in range(1, epochs + 1):
            train_loss, _ = _pass(model, loaders["train"], criterion, optimizer)
            val_loss, confusion = _pass(model, loaders["val"], criterion)
            scores = segmentation_metrics(confusion)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_miou": scores["miou"],
                    "val_dice": scores["dice"],
                    "val_pixel_acc": scores["pixel_acc"],
                }
            )
            best.offer(epoch, scores["miou"], model)

    with recorder.phase("evaluation"):
        model.load_state_dict(
            torch.load(best.path, map_location="cpu", weights_only=True)
        )
        predictions, per_image = _predict(model, loaders["test"])
        test_metrics = segmentation_metrics(np.array(per_image).sum(axis=0))

    with recorder.phase("evidence"):
        predictions_path = output / "test_predictions.npy"
        save_predictions(predictions_path, predictions)
        results: dict[str, Any] = {
            "device": "cpu",
            "seed": seed,
            "architecture": ARCHITECTURE,
            "num_classes": NUM_CLASSES,
            "ignore_index": IGNORE_INDEX,
            "image_size": image_size,
            "initial_weights_sha256": initial_hash,
            "history": history,
            "best_epoch": best.epoch,
            "best_val_miou": best.miou,
            "test_metrics": test_metrics,
            "test_stems": [image.stem for image, _ in pairs["test"]],
            "test_confusion_per_image": per_image,
            "test_predictions_file": str(predictions_path),
            "test_predictions_sha256": phase_contract.sha256_file(predictions_path),
            "best_model": str(best.path),
            "best_model_sha256": phase_contract.sha256_file(best.path),
        }
    return recorder.complete(results)
