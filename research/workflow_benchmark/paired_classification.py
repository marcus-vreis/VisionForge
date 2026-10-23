"""Check numerical parity before measuring the human workflow benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any


def run_visionforge(
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
    """Execute the baseline recipe through the public CPU Trainer and DataModule."""
    import torch
    from torch import nn
    from torchvision import datasets, models

    from visionforge.core.data import DataModule
    from visionforge.core.trainer import Trainer
    from visionforge.utils.config import ExperimentConfig

    if output.exists() or output.resolve().is_relative_to(dataset.resolve()):
        raise ValueError("Output must be new and outside the dataset")
    start = time.perf_counter()
    classes = datasets.ImageFolder(dataset / "train").classes
    cfg = ExperimentConfig.model_validate(
        {
            "name": "paired_cpu",
            "task": "multiclass",
            "model": {
                "name": "resnet18",
                "num_classes": len(classes),
                "pretrained": False,
            },
            "data": {
                "base_dir": str(dataset.resolve()),
                "num_workers": 0,
                "pin_memory": False,
                "transforms": {
                    "image_size": image_size,
                    "augment": False,
                    "normalize_mean": [0.5] * 3,
                    "normalize_std": [0.5] * 3,
                },
            },
            "training": {
                "epochs": epochs,
                "batch_size": batch_size,
                "seed": seed,
                "optimizer": "sgd",
                "learning_rate": learning_rate,
                "weight_decay": 0,
                "early_stopping_patience": 0,
                "mixed_precision": False,
                "deterministic": True,
                "scheduler": {"kind": "none"},
            },
            "device": {"kind": "cpu"},
            "output": {
                f"{name}_dir": str(output.resolve() / name)
                for name in ("models", "graphics", "logs", "reports")
            },
        }
    )
    data = DataModule(cfg)
    try:
        with torch.device("cpu"):
            model = models.resnet18(weights=None)
            classes = data.class_names
            model.fc = nn.Linear(model.fc.in_features, len(classes))
        model.load_state_dict(
            torch.load(initial_weights, map_location="cpu", weights_only=True)
        )
        setup_seconds = time.perf_counter() - start
        training_start = time.perf_counter()
        result = Trainer(cfg).fit(model, data)
        training_seconds = time.perf_counter() - training_start
        evaluation_start = time.perf_counter()
        model.load_state_dict(
            torch.load(result.model_path, map_location="cpu", weights_only=True)
        )
        model.eval()
        targets: list[int] = []
        predictions: list[int] = []
        with torch.no_grad():
            for images, labels in data.test_loader():
                targets.extend(labels.tolist())
                predictions.extend(model(images).argmax(dim=1).tolist())
        evaluation_seconds = time.perf_counter() - evaluation_start
        report = {
            "workflow": "visionforge_trainer",
            "device": "cpu",
            "seed": seed,
            "initial_weights_sha256": hashlib.sha256(
                initial_weights.read_bytes()
            ).hexdigest(),
            "class_to_idx": {name: index for index, name in enumerate(classes)},
            "history": [
                {
                    "epoch": row.epoch,
                    "train_loss": row.train_loss,
                    "val_loss": row.val_loss,
                    "train_acc": row.train_accuracy,
                    "val_acc": row.val_accuracy,
                }
                for row in result.history
            ],
            "best_epoch": result.best_epoch,
            "best_val_loss": result.best_val_loss,
            "test_accuracy": sum(
                a == b for a, b in zip(targets, predictions, strict=True)
            )
            / len(targets),
            "test_targets": targets,
            "test_predictions": predictions,
            "best_model": str(result.model_path),
            "setup_seconds": setup_seconds,
            "train_seconds": training_seconds,
            "evaluation_seconds": evaluation_seconds,
        }
        (output / "visionforge_result.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        return report
    finally:
        data.close()


def run_pair(
    dataset: Path,
    initial_weights: Path,
    output: Path,
    *,
    seed: int = 37,
    epochs: int = 2,
    batch_size: int = 4,
) -> dict[str, Any]:
    """Preserve both runs and require matching histories, predictions and weights."""
    import torch

    from visionforge.core.dataset_audit import audit_dataset
    from visionforge.core.dataset_fingerprint import fingerprint_dataset
    from visionforge.utils.environment import capture_environment

    from .native_classification import run_native

    dataset, initial_weights, output = (
        dataset.resolve(),
        initial_weights.resolve(),
        output.resolve(),
    )
    if output.is_relative_to(dataset) or dataset.is_relative_to(output):
        raise ValueError("Dataset and output must be separate directories")
    if output.exists():
        raise FileExistsError(output)
    audit = audit_dataset(dataset)
    if not audit["ok"]:
        raise ValueError("Dataset image audit has findings; inspect before pairing")
    fingerprint = fingerprint_dataset(dataset, method="content").to_dict()
    output.mkdir(parents=True)
    report: dict[str, Any] = {
        "purpose": "engineering_parity_only",
        "status": "running",
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "dataset_fingerprint": fingerprint,
        "environment": capture_environment(),
        "timing_scope": "Implementation phase observations; no speed or human-effort comparison is inferred.",
    }
    result_path = output / "paired_result.json"
    try:
        options = {"seed": seed, "epochs": epochs, "batch_size": batch_size}
        native = run_native(dataset, initial_weights, output / "native", **options)
        report["native"] = native
        platform = run_visionforge(
            dataset, initial_weights, output / "visionforge", **options
        )
        report["visionforge"] = platform
        compared = (
            "initial_weights_sha256",
            "class_to_idx",
            "history",
            "best_epoch",
            "best_val_loss",
            "test_accuracy",
            "test_targets",
            "test_predictions",
        )
        mismatches = [key for key in compared if native[key] != platform[key]]
        native_state = torch.load(
            native["best_model"], map_location="cpu", weights_only=True
        )
        platform_state = torch.load(
            platform["best_model"], map_location="cpu", weights_only=True
        )
        equal_weights = native_state.keys() == platform_state.keys() and all(
            torch.equal(value, platform_state[name])
            for name, value in native_state.items()
        )
        if not equal_weights:
            mismatches.append("best_weights")
        if fingerprint != fingerprint_dataset(dataset, method="content").to_dict():
            mismatches.append("dataset_changed")
        report.update(
            status="passed" if not mismatches else "failed", mismatches=mismatches
        )
        if mismatches:
            raise ValueError(f"Paired experiments differ: {', '.join(mismatches)}")
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        result_path.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    return report


def main() -> None:
    """Run an engineering pair using supplied data and initial weights, on CPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("initial_weights", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    result = run_pair(
        args.dataset,
        args.initial_weights,
        args.output,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "report": str(args.output / "paired_result.json"),
            }
        )
    )


if __name__ == "__main__":
    # Must precede the first torch import in CLI use.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    main()
