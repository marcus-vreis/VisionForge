"""Check numerical parity before measuring the human workflow benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import phase_contract

VISIONFORGE_INCLUDES = {
    "setup": [
        "ExperimentConfig validation",
        "DataModule ImageFolder indexing of train/val/test",
        "ResNet18 construction and initial-weights load",
        "initial-weights SHA-256",
        "Trainer construction",
    ],
    "training": [
        "Trainer.fit: seeding, optimizer and loss construction",
        "per-epoch training and validation passes",
        "best-checkpoint writes",
        "per-epoch resume state, TensorBoard logs, health checks and run.json",
    ],
    "evaluation": ["best-checkpoint reload", "test pass with predictions"],
    "evidence": ["best-checkpoint SHA-256", "result assembly"],
}
EXECUTION_ORDER = ["native_pytorch", "visionforge_trainer"]


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
    """Execute the baseline recipe through the public CPU Trainer and DataModule."""
    import torch
    from torch import nn
    from torchvision import models

    from visionforge.core.data import DataModule
    from visionforge.core.trainer import Trainer
    from visionforge.utils.config import ExperimentConfig

    from .native_classification import preflight

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
        "visionforge_trainer",
        output / "visionforge_result.json",
        VISIONFORGE_INCLUDES,
        phase_contract.capture_conditions(concurrent_workload),
    )
    data = None
    try:
        with recorder.phase("setup"):
            cfg = ExperimentConfig.model_validate(
                _config(
                    dataset,
                    output,
                    len(class_to_idx),
                    seed,
                    epochs,
                    batch_size,
                    image_size,
                    learning_rate,
                )
            )
            data = DataModule(cfg)
            classes = data.class_names
            with torch.device("cpu"):
                model = models.resnet18(weights=None)
                model.fc = nn.Linear(model.fc.in_features, len(classes))
            model.load_state_dict(
                torch.load(initial_weights, map_location="cpu", weights_only=True)
            )
            initial_hash = phase_contract.sha256_file(initial_weights)
            trainer = Trainer(cfg)

        with recorder.phase("training"):
            result = trainer.fit(model, data)

        targets: list[int] = []
        predictions: list[int] = []
        with recorder.phase("evaluation"):
            model.load_state_dict(
                torch.load(result.model_path, map_location="cpu", weights_only=True)
            )
            model.eval()
            with torch.inference_mode():
                for images, labels in data.test_loader():
                    targets.extend(labels.tolist())
                    predictions.extend(model(images).argmax(dim=1).tolist())

        with recorder.phase("evidence"):
            results = {
                "device": "cpu",
                "seed": seed,
                "initial_weights_sha256": initial_hash,
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
                "best_model_sha256": phase_contract.sha256_file(result.model_path),
            }
        return recorder.complete(results)
    finally:
        if data is not None:
            data.close()


def _config(
    dataset: Path,
    output: Path,
    num_classes: int,
    seed: int,
    epochs: int,
    batch_size: int,
    image_size: int,
    learning_rate: float,
) -> dict[str, Any]:
    """Build the restricted recipe as an ExperimentConfig payload."""
    return {
        "name": "paired_cpu",
        "task": "multiclass",
        "model": {"name": "resnet18", "num_classes": num_classes, "pretrained": False},
        "data": {
            "base_dir": str(dataset),
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
            f"{name}_dir": str(output / name)
            for name in ("models", "graphics", "logs", "reports")
        },
    }


def run_pair(
    dataset: Path,
    initial_weights: Path,
    output: Path,
    *,
    seed: int = 37,
    epochs: int = 2,
    batch_size: int = 4,
    concurrent_workload: str = "undeclared",
) -> dict[str, Any]:
    """Preserve both runs and require matching histories, predictions and weights."""
    import torch

    from visionforge.core.dataset_audit import audit_dataset
    from visionforge.core.dataset_fingerprint import fingerprint_dataset
    from visionforge.utils.environment import capture_environment

    from .native_classification import preflight, run_native

    dataset, initial_weights, output = (
        dataset.resolve(),
        initial_weights.resolve(),
        output.resolve(),
    )
    if output.is_relative_to(dataset) or dataset.is_relative_to(output):
        raise ValueError("Dataset and output must be separate directories")
    if output.exists():
        raise FileExistsError(output)
    preflight(
        dataset,
        initial_weights,
        output,
        epochs=epochs,
        batch_size=batch_size,
        image_size=32,
        learning_rate=0.01,
    )
    audit = audit_dataset(dataset)
    if not audit["ok"]:
        raise ValueError("Dataset image audit has findings; inspect before pairing")
    fingerprint = fingerprint_dataset(dataset, method="content").to_dict()
    output.mkdir(parents=True)
    report: dict[str, Any] = {
        "purpose": "engineering_parity_only",
        "status": "running",
        "failed_phase": None,
        "phase_contract": phase_contract.contract_header(),
        "execution_order": EXECUTION_ORDER,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "concurrent_workload": concurrent_workload,
        "dataset_fingerprint": fingerprint,
        "environment": capture_environment(),
    }
    options: dict[str, Any] = {
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "concurrent_workload": concurrent_workload,
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
    parser.add_argument("--batch-size", type=int, default=4)
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
