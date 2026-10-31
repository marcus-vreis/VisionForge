"""Independent SSDLite training; only the mAP evaluator is shared with VisionForge."""

from __future__ import annotations

if __name__ == "__main__":
    from .paired_detection import main

    main(native=True)
    raise SystemExit

import hashlib
import json
import math
import random
from functools import partial
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import _utils, ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from torchvision.transforms.functional import to_tensor

from . import phase_contract
from .prepare_voc import CLASSES

ARCHITECTURE = "ssdlite320_mobilenet_v3_large"
SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
NATIVE_INCLUDES = {
    "setup": [
        "YOLO loaders (workers=0)",
        "SSDLite construction, initial weights load and device placement",
        "initial weights SHA-256",
    ],
    "training": [
        "seeding and deterministic algorithms",
        "SGD construction after device placement",
        "train, validation loss in train mode (including BatchNorm updates), validation mAP in eval mode",
        "best checkpoint CPU state writes",
    ],
    "evaluation": [
        "best checkpoint reload",
        "test or first available split using shared all-points mAP@50 evaluator",
        "CPU predictions in sorted image order",
    ],
    "evidence": [
        "full float32 predictions JSON write",
        "SHA-256 of predictions rounded to 1e-6",
        "checkpoint SHA-256 and result assembly",
    ],
}


def build_detector(num_classes):
    """Build SSDLite independently with the same head replacement as the factory."""
    model = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None)
    channels = _utils.retrieve_out_channels(model.backbone, (320, 320))
    anchors = model.anchor_generator.num_anchors_per_location()
    model.head.classification_head = SSDLiteClassificationHead(
        channels,
        anchors,
        num_classes + 1,
        partial(nn.BatchNorm2d, eps=1e-3, momentum=0.03),
    )
    return model


def make_initial_weights(path, seed, num_classes=20):
    """Save a new CPU state dictionary without pretrained downloads."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    torch.random.default_generator.manual_seed(seed)
    with torch.device("cpu"):
        model = build_detector(num_classes)
    with path.open("xb") as stream:
        torch.save(model.state_dict(), stream)


class YoloDataset(Dataset):
    """Read YOLO labels independently, retaining torchvision background label zero."""

    def __init__(self, split):
        self.split = Path(split)
        self.images = sorted(
            p
            for p in (self.split / "images").iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        with Image.open(path) as source:
            image = source.convert("RGB")
            width, height = image.size
            tensor = to_tensor(image)
        boxes, labels = [], []
        for line in (
            (self.split / "labels" / f"{path.stem}.txt")
            .read_text(encoding="utf-8")
            .splitlines()
        ):
            if not line.strip():
                continue
            cls, cx, cy, w, h = map(float, line.split())
            x1, y1 = (cx - w / 2) * width, (cy - h / 2) * height
            x2, y2 = (cx + w / 2) * width, (cy + h / 2) * height
            x1, x2 = max(0.0, x1), min(float(width), x2)
            y1, y2 = max(0.0, y1), min(float(height), y2)
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(int(cls) + 1)
        return tensor, {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }


def collate(batch):
    """Return parallel tuples of images and targets."""
    return tuple(zip(*batch, strict=True))


def class_names(dataset):
    """Use manifest classes, classes.txt, or the fixed VOC vocabulary."""
    manifest = dataset / "manifest.json"
    if manifest.is_file():
        names = json.loads(manifest.read_text(encoding="utf-8"))["classes"]
    elif (dataset / "classes.txt").is_file():
        names = (dataset / "classes.txt").read_text(encoding="utf-8").splitlines()
    else:
        names = list(CLASSES)
    if (
        not names
        or any(not isinstance(n, str) or not n for n in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("Invalid class vocabulary")
    return names


def preflight(
    dataset,
    initial_weights,
    output,
    *,
    seed,
    epochs,
    batch_size,
    learning_rate,
    momentum=0.937,
    weight_decay=5e-4,
):
    """Validate controls and YOLO labels without creating output."""
    if not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be at least 1")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(momentum) or not 0 <= momentum <= 1:
        raise ValueError("momentum must be finite and in [0, 1]")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if output.is_relative_to(dataset) or dataset.is_relative_to(output):
        raise ValueError("Dataset and output must be separate directories")
    if output.exists():
        raise FileExistsError(output)
    if initial_weights.is_relative_to(output):
        raise ValueError("Initial weights must be outside output")
    if not initial_weights.is_file():
        raise FileNotFoundError(initial_weights)
    names = class_names(dataset)
    splits = {}
    for split in SPLITS:
        directory = dataset / split
        if (dataset / "images" / split).exists():
            raise ValueError(
                "Only the unambiguous <split>/images YOLO layout is supported"
            )
        if split == "test" and not directory.exists():
            continue
        if not (directory / "images").is_dir() or not (directory / "labels").is_dir():
            raise ValueError(f"{split} requires images/ and labels/")
        ds = YoloDataset(directory)
        stems = [p.stem for p in ds.images]
        if not stems or len(set(stems)) != len(stems):
            raise ValueError(f"Empty split or duplicate image stems: {split}")
        if set(stems) != {p.stem for p in (directory / "labels").glob("*.txt")}:
            raise ValueError(f"Unpaired images and labels: {split}")
        for stem in stems:
            for line in (
                (directory / "labels" / f"{stem}.txt")
                .read_text(encoding="utf-8")
                .splitlines()
            ):
                if not line.strip():
                    continue
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError("YOLO labels require five columns")
                cls, *box = map(float, fields)
                if (
                    not all(math.isfinite(v) for v in (cls, *box))
                    or not cls.is_integer()
                    or not 0 <= cls < len(names)
                ):
                    raise ValueError("Invalid YOLO class or nonfinite box")
        # Train-mode validation also updates BatchNorm; its last batch needs two.
        if split in ("train", "val") and (
            batch_size == 1 or len(ds) % batch_size == 1 or len(ds) == 1
        ):
            raise ValueError(
                "Train and validation batches need at least two images for BatchNorm"
            )
        splits[split] = ds
    return names, splits


def loader(ds, batch_size, *, shuffle=False):
    """Construct a zero-worker loader without consuming an iterator."""
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate,
    )


def evaluate_detector(model, loader, device):
    """Return shared VisionForge all-points mAP@50 and CPU predictions."""
    from visionforge.core.detection_metrics import mean_average_precision_50

    device = "cuda:0" if device == "cuda" else device
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for images, truth in loader:
            outputs = model([image.to(device) for image in images])
            predictions.extend(
                {k: v.detach().cpu() for k, v in out.items()} for out in outputs
            )
            targets.extend(
                {"boxes": t["boxes"].cpu(), "labels": t["labels"].cpu()} for t in truth
            )
    return mean_average_precision_50(predictions, targets).map50, predictions


def _loss_pass(model, loader, device, optimizer=None):
    device = "cuda:0" if device == "cuda" else device
    model.train()
    total, count = 0.0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for images, targets in loader:
            images = [image.to(device) for image in images]
            targets = [
                {k: v.to(device) for k, v in target.items()} for target in targets
            ]
            loss = sum(model(images, targets).values())
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total += float(loss.item())
            count += 1
    return total / count if count else 0.0


def save_predictions(path, predictions):
    """Save full float32 values; hash canonical JSON rounded to six decimals."""
    full = [
        {key: row[key].tolist() for key in ("boxes", "scores", "labels")}
        for row in predictions
    ]
    rounded = [
        {
            key: np.round(row[key].numpy().astype(np.float64), 6).tolist()
            if key != "labels"
            else row[key].tolist()
            for key in ("boxes", "scores", "labels")
        }
        for row in predictions
    ]
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(full, stream, allow_nan=False)
    return hashlib.sha256(
        json.dumps(
            rounded, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def evidence(
    model_path,
    predictions,
    output,
    *,
    names,
    history,
    best_epoch,
    test_map50,
    initial_hash,
    device,
    parity_atol,
    seed,
    test_split,
    stems,
):
    """Assemble the common numerical evidence inside the evidence phase."""
    path = output / "test_predictions.json"
    digest = save_predictions(path, predictions)
    return {
        "architecture": ARCHITECTURE,
        "num_classes": len(names),
        "class_mapping": {name: i for i, name in enumerate(names)},
        "seed": seed,
        "device": device,
        "parity_atol": parity_atol,
        "initial_weights_sha256": initial_hash,
        "history": history,
        "best_epoch": best_epoch,
        "test_map50": test_map50,
        "test_split": test_split,
        "test_stems": stems,
        "test_predictions_file": str(path),
        "test_predictions_sha256": digest,
        "prediction_hash_encoding": "canonical sorted-key compact JSON, floats rounded to 1e-6; file preserves complete float32 values",
        "best_model": str(model_path),
        "best_model_sha256": phase_contract.sha256_file(model_path),
    }


def run_native(
    dataset,
    initial_weights,
    output,
    *,
    seed,
    epochs,
    batch_size,
    learning_rate=0.01,
    momentum=0.937,
    weight_decay=5e-4,
    concurrent_workload="undeclared",
    device="cpu",
    parity_atol=None,
):
    """Run native detection under the shared setup/training/evaluation/evidence contract."""
    parity_atol = phase_contract.validate_device(device, parity_atol)
    dataset, initial_weights, output = (
        Path(p).resolve() for p in (dataset, initial_weights, output)
    )
    names, splits = preflight(
        dataset,
        initial_weights,
        output,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    output.mkdir(parents=True)
    recorder = phase_contract.PhaseRecorder(
        "native_pytorch",
        output / "native_result.json",
        NATIVE_INCLUDES,
        phase_contract.capture_conditions(concurrent_workload, device),
    )
    with recorder.phase("setup"):
        model = build_detector(len(names))
        model.load_state_dict(
            torch.load(initial_weights, map_location="cpu", weights_only=True)
        )
        model.to("cuda:0" if device == "cuda" else device)
        loaders = {
            split: loader(ds, batch_size, shuffle=split == "train")
            for split, ds in splits.items()
        }
        initial_hash = phase_contract.sha256_file(initial_weights)
    history, best_map, best_epoch = [], -1.0, 0
    best_path = output / "best_model.pth"
    with recorder.phase("training"):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=device == "cuda")
        optimizer = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad],
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        for epoch in range(1, epochs + 1):
            train_loss = _loss_pass(model, loaders["train"], device, optimizer)
            val_loss = _loss_pass(model, loaders["val"], device)
            map50, _ = evaluate_detector(model, loaders["val"], device)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_map50": map50,
                }
            )
            if map50 > best_map:
                best_map, best_epoch = map50, epoch
                torch.save(
                    {
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    },
                    best_path,
                )
        if history and not best_path.exists():
            torch.save(
                {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                best_path,
            )
    with recorder.phase("evaluation"):
        model.load_state_dict(
            torch.load(best_path, map_location="cpu", weights_only=True)
        )
        test_split = "test" if "test" in loaders else next(iter(loaders))
        # Evaluation must be sorted even when falling back to the training split.
        test_map50, predictions = evaluate_detector(
            model, loader(splits[test_split], batch_size), device
        )
    with recorder.phase("evidence"):
        result = evidence(
            best_path,
            predictions,
            output,
            names=names,
            history=history,
            best_epoch=best_epoch,
            test_map50=test_map50,
            initial_hash=initial_hash,
            device=device,
            parity_atol=parity_atol,
            seed=seed,
            test_split=test_split,
            stems=[p.stem for p in splits[test_split].images],
        )
    return recorder.complete(result)
