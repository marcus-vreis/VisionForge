"""Check segmentation numerical parity before measuring the workflow benchmark (ADR-149)."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import phase_contract
from .trimap_labels import IGNORE_INDEX, TRIMAP_TO_MASK

# The platform side states the recipe itself, so a drift on either side shows up.
RECIPE_NORMALIZE_MEAN = [0.5, 0.5, 0.5]
RECIPE_NORMALIZE_STD = [0.5, 0.5, 0.5]
RECIPE_NUM_CLASSES = 2
# NumPy and torch float64 reductions over identical integer confusion matrices
# differ by rounding only; one changed pixel moves a metric by >= 1/labelled pixels.
METRIC_ABS_TOLERANCE = 1e-12
TOLERANCES = {
    "losses": "exact",
    "best_epoch": "exact",
    "best_weights": "exact",
    "test_predictions": "exact",
    "test_confusion_per_image": "exact",
    "metrics_abs": METRIC_ABS_TOLERANCE,
}
_METRIC_KEYS = ("val_miou", "val_dice", "val_pixel_acc")
EXECUTION_ORDER = ["native_pytorch", "visionforge_trainer"]
VISIONFORGE_INCLUDES = {
    "setup": [
        "SegmentationConfig validation",
        "SegmentationDataModule image/mask pairing of train/val/test",
        "tiny_fcn construction and initial-weights load",
        "initial-weights SHA-256",
        "SegmentationTrainer construction",
    ],
    "training": [
        "SegmentationTrainer.fit: seeding, optimizer and loss construction",
        "per-epoch training and validation passes with mask-id checks",
        "best-checkpoint writes",
        "per-epoch resume state, TensorBoard logs, health checks and run.json",
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


def _config(
    dataset: Path,
    output: Path,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    image_size: int,
    learning_rate: float,
) -> dict[str, Any]:
    """Build the restricted recipe as a SegmentationConfig payload."""
    return {
        "name": "paired_segmentation_cpu",
        "task": "segmentation",
        # Validation only: the harness passes its own tiny_fcn to fit().
        "model": {
            "name": "unet",
            "num_classes": RECIPE_NUM_CLASSES,
            "pretrained": False,
        },
        "data": {
            "base_dir": str(dataset),
            "image_size": image_size,
            "ignore_index": IGNORE_INDEX,
            "num_workers": 0,
            "pin_memory": False,
            "transforms": {
                "image_size": image_size,
                "augment": False,
                "horizontal_flip": False,
                "rotation_degrees": 0,
                "color_jitter": False,
                "normalize_mean": RECIPE_NORMALIZE_MEAN,
                "normalize_std": RECIPE_NORMALIZE_STD,
            },
        },
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "seed": seed,
            "optimizer": "sgd",
            "learning_rate": learning_rate,
            "weight_decay": 0,
            "loss": "cross_entropy",
            "early_stopping_patience": 0,
            "deterministic": True,
            "scheduler": {"kind": "none"},
        },
        "device": {"kind": "cpu"},
        "output": {
            f"{name}_dir": str(output / name)
            for name in ("models", "graphics", "logs", "reports")
        },
    }


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
    concurrent_workload: str = "undeclared",
) -> dict[str, Any]:
    """Execute the recipe through the public SegmentationDataModule and Trainer, on CPU."""
    import numpy as np
    import torch

    from visionforge.core.segmentation_data import SegmentationDataModule
    from visionforge.core.segmentation_trainer import (
        SegmentationMetricAccumulator,
        SegmentationTrainer,
        per_image_confusion,
    )
    from visionforge.models.segmentation_factory import segmentation_logits
    from visionforge.utils.segmentation_config import SegmentationConfig

    from .native_segmentation import (
        ARCHITECTURE,
        build_tiny_fcn,
        preflight,
        save_predictions,
    )

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
        "visionforge_trainer",
        output / "visionforge_result.json",
        VISIONFORGE_INCLUDES,
        phase_contract.capture_conditions(concurrent_workload),
    )
    data = None
    try:
        with recorder.phase("setup"):
            cfg = SegmentationConfig.model_validate(
                _config(
                    dataset,
                    output,
                    seed=seed,
                    epochs=epochs,
                    batch_size=batch_size,
                    image_size=image_size,
                    learning_rate=learning_rate,
                )
            )
            data = SegmentationDataModule(cfg)
            model = build_tiny_fcn(cfg.model.num_classes)
            model.load_state_dict(
                torch.load(initial_weights, map_location="cpu", weights_only=True)
            )
            initial_hash = phase_contract.sha256_file(initial_weights)
            trainer = SegmentationTrainer(cfg)

        with recorder.phase("training"):
            result = trainer.fit(model, data)

        num_classes = cfg.model.num_classes
        with recorder.phase("evaluation"):
            model.load_state_dict(
                torch.load(result.model_path, map_location="cpu", weights_only=True)
            )
            model.eval()
            metrics = SegmentationMetricAccumulator(num_classes, IGNORE_INDEX)
            per_image: list[Any] = []
            predictions: list[Any] = []
            loader = data.test_loader()
            if loader is None:
                raise ValueError("test split is required")
            with torch.inference_mode():
                for images, masks in loader:
                    logits = segmentation_logits(model(images))
                    metrics.update(logits, masks)
                    per_image.append(
                        per_image_confusion(logits, masks, num_classes, IGNORE_INDEX)
                    )
                    predictions.append(logits.argmax(dim=1).numpy().astype(np.uint8))
            miou, dice, pixel_acc = metrics.compute()

        with recorder.phase("evidence"):
            predictions_path = output / "test_predictions.npy"
            save_predictions(predictions_path, np.concatenate(predictions))
            results: dict[str, Any] = {
                "device": "cpu",
                "seed": seed,
                "architecture": ARCHITECTURE,
                "num_classes": num_classes,
                "ignore_index": IGNORE_INDEX,
                "image_size": image_size,
                "initial_weights_sha256": initial_hash,
                "history": [
                    {
                        "epoch": row.epoch,
                        "train_loss": row.train_loss,
                        "val_loss": row.val_loss,
                        "val_miou": row.val_miou,
                        "val_dice": row.val_dice,
                        "val_pixel_acc": row.val_pixel_acc,
                    }
                    for row in result.history
                ],
                "best_epoch": result.best_epoch,
                "best_val_miou": result.best_val_miou,
                "test_metrics": {"miou": miou, "dice": dice, "pixel_acc": pixel_acc},
                "test_stems": [image.stem for image, _ in pairs["test"]],
                "test_confusion_per_image": np.concatenate(per_image).tolist(),
                "test_predictions_file": str(predictions_path),
                "test_predictions_sha256": phase_contract.sha256_file(predictions_path),
                "best_model": str(result.model_path),
                "best_model_sha256": phase_contract.sha256_file(result.model_path),
            }
        return recorder.complete(results)
    finally:
        if data is not None:
            data.close()


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= METRIC_ABS_TOLERANCE


def _history_matches(
    native: list[Mapping[str, Any]], platform: list[Mapping[str, Any]]
) -> bool:
    if len(native) != len(platform):
        return False
    for left, right in zip(native, platform, strict=True):
        if left.keys() != right.keys():
            return False
        for key, value in left.items():
            same = (
                _close(value, right[key])
                if key in _METRIC_KEYS
                else value == right[key]
            )
            if not same:
                return False
    return True


def compare_results(
    native: Mapping[str, Any], platform: Mapping[str, Any]
) -> list[str]:
    """Return the names of compared fields that differ beyond the ADR-149 tolerances."""
    mismatches: list[str] = []
    if native["initial_weights_sha256"] != platform["initial_weights_sha256"]:
        mismatches.append("initial_weights_sha256")
    if not _history_matches(native["history"], platform["history"]):
        mismatches.append("history")
    if native["best_epoch"] != platform["best_epoch"]:
        mismatches.append("best_epoch")
    if not _close(native["best_val_miou"], platform["best_val_miou"]):
        mismatches.append("best_val_miou")
    left, right = native["test_metrics"], platform["test_metrics"]
    if left.keys() != right.keys() or not all(_close(left[k], right[k]) for k in left):
        mismatches.append("test_metrics")
    if native["test_confusion_per_image"] != platform["test_confusion_per_image"]:
        mismatches.append("test_confusion_per_image")
    if native["test_predictions_sha256"] != platform["test_predictions_sha256"]:
        mismatches.append("test_predictions")
    return mismatches


def _same_weights(native_path: str, platform_path: str) -> bool:
    import torch

    native = torch.load(native_path, map_location="cpu", weights_only=True)
    platform = torch.load(platform_path, map_location="cpu", weights_only=True)
    return native.keys() == platform.keys() and all(
        torch.equal(value, platform[name]) for name, value in native.items()
    )


def run_pair(
    dataset: Path,
    initial_weights: Path,
    output: Path,
    *,
    seed: int = 37,
    epochs: int = 2,
    batch_size: int = 2,
    image_size: int = 32,
    learning_rate: float = 0.01,
    concurrent_workload: str = "undeclared",
) -> dict[str, Any]:
    """Preserve both runs and require matching histories, predictions and weights."""
    from visionforge.core.dataset_fingerprint import fingerprint_dataset
    from visionforge.utils.environment import capture_environment

    from .native_segmentation import run_native

    dataset, initial_weights, output = (
        Path(dataset).resolve(),
        Path(initial_weights).resolve(),
        Path(output).resolve(),
    )
    if output.is_relative_to(dataset) or dataset.is_relative_to(output):
        raise ValueError("Dataset and output must be separate directories")
    if output.exists():
        raise FileExistsError(output)
    fingerprint = fingerprint_dataset(dataset, method="content").to_dict()
    output.mkdir(parents=True)
    options: dict[str, Any] = {
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "image_size": image_size,
        "learning_rate": learning_rate,
        "concurrent_workload": concurrent_workload,
    }
    report: dict[str, Any] = {
        "purpose": "engineering_parity_only",
        "task": "segmentation",
        "status": "running",
        "failed_phase": None,
        "phase_contract": phase_contract.contract_header(),
        "execution_order": EXECUTION_ORDER,
        **options,
        "tolerances": TOLERANCES,
        "label_mapping": {
            "mask": {"background": 0, "foreground": 1, "ignore": IGNORE_INDEX},
            "oxford_trimap": {str(k): v for k, v in TRIMAP_TO_MASK.items()},
        },
        "dataset_fingerprint": fingerprint,
        "environment": capture_environment(),
    }
    runners = {
        "native_pytorch": ("native", run_native),
        "visionforge_trainer": ("visionforge", run_visionforge),
    }
    try:
        for workflow in EXECUTION_ORDER:
            key, runner = runners[workflow]
            try:
                report[key] = runner(dataset, initial_weights, output / key, **options)
            except phase_contract.PhaseFailure as failure:
                report.update(
                    {key: failure.report, "failed_phase": f"{key}.{failure.phase}"}
                )
                raise
        report["failed_phase"] = "comparison"
        native, platform = report["native"], report["visionforge"]
        mismatches = compare_results(native, platform)
        if not _same_weights(native["best_model"], platform["best_model"]):
            mismatches.append("best_weights")
        if fingerprint != fingerprint_dataset(dataset, method="content").to_dict():
            mismatches.append("dataset_changed")
        report.update(
            status="passed" if not mismatches else "failed",
            mismatches=mismatches,
            failed_phase="comparison" if mismatches else None,
        )
        if mismatches:
            raise ValueError(f"Paired experiments differ: {', '.join(mismatches)}")
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        phase_contract.write_report_once(output / "paired_result.json", report)
    return report


def main() -> None:
    """Run an engineering pair using supplied data and initial weights, on CPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("initial_weights", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument(
        "--concurrent-workload",
        default="undeclared",
        help="What else ran on this machine; recorded in every report.",
    )
    args = parser.parse_args()
    result = run_pair(
        args.dataset,
        args.initial_weights,
        args.output,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        learning_rate=args.learning_rate,
        concurrent_workload=args.concurrent_workload,
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
    phase_contract.prepare_cpu_process()
    main()
