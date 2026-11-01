"""Check native SSDLite versus the public VisionForge DetectionTrainer."""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path

from . import phase_contract

METRIC_ABS_TOLERANCE = 1e-12
EXECUTION_ORDER = ["native_pytorch", "visionforge_trainer"]
VISIONFORGE_INCLUDES = {
    "setup": [
        "DetectionConfig validation (torchvision, SSDLite, pretrained=False)",
        "model construction, shared initial weights load and device placement",
        "DetectionTrainer construction and initial weights SHA-256",
    ],
    "training": [
        "DetectionTrainer.fit: seeding, injected prebuilt model, SGD and YOLO loaders",
        "train, validation loss (train mode including BatchNorm updates), validation mAP",
        "best checkpoint, resume state, TensorBoard, plots, health checks and run.json",
        "public epoch_end callback captures train and validation losses",
    ],
    "evaluation": [
        "best.pt reload",
        "test or first available split using shared all-points mAP@50 evaluator",
        "CPU predictions in sorted image order",
    ],
    "evidence": [
        "full float32 predictions JSON write",
        "SHA-256 of predictions rounded to 1e-6",
        "checkpoint SHA-256 and result assembly",
    ],
}


@contextmanager
def inject_initial_model(model):
    """Temporarily replace only the trainer's factory; consume no RNG in fit."""
    import visionforge.core.detection_trainer as module

    original = module.build_torchvision_detector
    module.build_torchvision_detector = lambda *args, **kwargs: model
    try:
        yield
    finally:
        module.build_torchvision_detector = original


def _config(
    dataset,
    output,
    *,
    names,
    seed,
    epochs,
    batch_size,
    learning_rate,
    momentum,
    weight_decay,
    device,
):
    return {
        "name": f"paired_detection_{device}",
        "model": {
            "backend": "torchvision",
            "name": "ssdlite320_mobilenet_v3_large",
            "num_classes": len(names),
            "pretrained": False,
        },
        "data": {"base_dir": str(dataset), "class_names": names, "image_size": 320},
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "optimizer": "SGD",
            "momentum": momentum,
            "weight_decay": weight_decay,
            "workers": 0,
            "seed": seed,
            "deterministic": True,
            "amp": False,
            "patience": 0,
            "warmup_epochs": 0,
            "augmentation": {"augment": False},
        },
        "device": {"kind": device},
        "output": {
            f"{name}_dir": str(output / name)
            for name in ("models", "graphics", "logs", "reports")
        },
    }


def run_visionforge(
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
    """Train through DetectionTrainer.fit with a scoped initial-weights injection."""
    import torch

    from visionforge.core.detection_trainer import DetectionTrainer
    from visionforge.utils.detection_config import DetectionConfig

    from .native_detection import (
        build_detector,
        evaluate_detector,
        evidence,
        loader,
        preflight,
    )

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
        "visionforge_trainer",
        output / "visionforge_result.json",
        VISIONFORGE_INCLUDES,
        phase_contract.capture_conditions(concurrent_workload, device),
    )
    with recorder.phase("setup"):
        cfg = DetectionConfig.model_validate(
            _config(
                dataset,
                output,
                names=names,
                seed=seed,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
                device=device,
            )
        )
        model = build_detector(len(names))
        model.load_state_dict(
            torch.load(initial_weights, map_location="cpu", weights_only=True)
        )
        model.to("cuda:0" if device == "cuda" else device)
        initial_hash = phase_contract.sha256_file(initial_weights)
        trainer = DetectionTrainer(cfg)
    history = []

    def progress(event):
        if event["event"] == "epoch_end":
            history.append(
                {
                    "epoch": event["epoch"],
                    "train_loss": event["train_loss"],
                    "val_loss": event["val_loss"],
                    "val_map50": event["map50"],
                }
            )

    with recorder.phase("training"):
        torch.use_deterministic_algorithms(True, warn_only=device == "cuda")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        with inject_initial_model(model):
            result = trainer.fit(progress_callback=progress)
        if next(model.parameters()).device.type != device:
            raise RuntimeError(f"Trainer did not use requested device {device}")
    with recorder.phase("evaluation"):
        best_path = result.run_dir / "weights" / "best.pt"
        model.load_state_dict(
            torch.load(best_path, map_location="cpu", weights_only=True)
        )
        test_split = "test" if "test" in splits else next(iter(splits))
        test_map50, predictions = evaluate_detector(
            model, loader(splits[test_split], batch_size), device
        )
    with recorder.phase("evidence"):
        results = evidence(
            best_path,
            predictions,
            output,
            names=names,
            history=history,
            best_epoch=result.best_epoch,
            test_map50=test_map50,
            initial_hash=initial_hash,
            device=device,
            parity_atol=parity_atol,
            seed=seed,
            test_split=test_split,
            stems=[p.stem for p in splits[test_split].images],
        )
        results["initial_weights_injection"] = (
            "monkeypatch of detection_trainer.build_torchvision_detector"
        )
        results["config"] = cfg.model_dump(mode="json")
        results["config_notes"] = (
            "DetectionConfig is standalone; image_size=320, amp=False, patience=0, warmup_epochs=0, augmentation disabled; torchvision owns resize/normalization and ignores Ultralytics-only schedule/loss-gain fields"
        )
    return recorder.complete(results)


def _predictions_match(left, right, atol):
    import numpy as np

    if len(left) != len(right):
        return False
    for a, b in zip(left, right, strict=True):
        for key in ("boxes", "scores", "labels"):
            x, y = np.asarray(a[key]), np.asarray(b[key])
            if x.shape != y.shape:
                return False
            if key == "labels" or atol == 0:
                if not np.array_equal(x, y):
                    return False
            elif not np.allclose(x, y, atol=atol, rtol=0, equal_nan=False):
                return False
    return True


def compare_results(native, platform, loss_atol=0.0, *, device="cpu"):
    """Compare exact metadata, bounded losses/metrics and complete predictions."""
    mismatches = []
    for key in (
        "initial_weights_sha256",
        "class_mapping",
        "num_classes",
        "best_epoch",
        "test_split",
        "test_stems",
    ):
        if native[key] != platform[key]:
            mismatches.append(key)
    a, b = native["history"], platform["history"]
    same = len(a) == len(b)
    for left, right in zip(a, b, strict=False):
        if left.keys() != right.keys():
            same = False
            continue
        for key in left:
            tolerance = (
                METRIC_ABS_TOLERANCE
                if key == "val_map50"
                else loss_atol
                if key in ("train_loss", "val_loss")
                else 0
            )
            if not abs(left[key] - right[key]) <= tolerance:
                same = False
    if not same:
        mismatches.append("history")
    if not abs(native["test_map50"] - platform["test_map50"]) <= METRIC_ABS_TOLERANCE:
        mismatches.append("test_map50")
    if (
        device == "cpu"
        and native["test_predictions_sha256"] != platform["test_predictions_sha256"]
    ):
        mismatches.append("test_predictions")
    left = json.loads(Path(native["test_predictions_file"]).read_text(encoding="utf-8"))
    right = json.loads(
        Path(platform["test_predictions_file"]).read_text(encoding="utf-8")
    )
    if not _predictions_match(left, right, loss_atol if device == "cuda" else 0):
        if "test_predictions" not in mismatches:
            mismatches.append("test_predictions")
    return mismatches


def fingerprint_dataset(dataset):
    """Hash all relative paths and file contents, including YOLO labels and metadata."""
    from visionforge.core.dataset_fingerprint import fingerprint_dataset as fingerprint

    result = fingerprint(dataset, method="content").to_dict()
    if result["method"] != "content":
        raise ValueError(f"Dataset fingerprint unavailable: {result['note']}")
    return result


def run_pair(
    dataset,
    initial_weights,
    output,
    *,
    seed=37,
    epochs=2,
    batch_size=2,
    learning_rate=0.01,
    momentum=0.937,
    weight_decay=5e-4,
    concurrent_workload="undeclared",
    device="cpu",
    parity_atol=None,
):
    """Run native first and retain paired evidence even when a phase fails."""
    from visionforge.utils.environment import capture_environment

    from .native_detection import preflight, run_native

    parity_atol = phase_contract.validate_device(device, parity_atol)
    dataset, initial_weights, output = (
        Path(p).resolve() for p in (dataset, initial_weights, output)
    )
    options = {
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "concurrent_workload": concurrent_workload,
        "device": device,
        "parity_atol": parity_atol,
    }
    preflight(
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
    fingerprint = fingerprint_dataset(dataset)
    output.mkdir(parents=True)
    report = {
        "purpose": "engineering_parity_only",
        "task": "detection",
        "status": "running",
        "failed_phase": None,
        "phase_contract": phase_contract.contract_header(),
        "execution_order": EXECUTION_ORDER,
        **options,
        "tolerances": {
            "losses": {"atol": parity_atol, "rtol": 0},
            "metrics_abs": METRIC_ABS_TOLERANCE,
            "best_weights": {"atol": parity_atol, "rtol": 0},
            "test_predictions": "exact arrays on CPU; boxes/scores atol=parity_atol on CUDA; labels exact",
        },
        "dataset_fingerprint": fingerprint,
        "environment": capture_environment(),
    }
    try:
        for key, runner in (("native", run_native), ("visionforge", run_visionforge)):
            try:
                report[key] = runner(dataset, initial_weights, output / key, **options)
            except phase_contract.PhaseFailure as failure:
                report.update(
                    {key: failure.report, "failed_phase": f"{key}.{failure.phase}"}
                )
                raise
        report["failed_phase"] = "comparison"
        native, platform = report["native"], report["visionforge"]
        mismatches = compare_results(native, platform, parity_atol, device=device)
        import torch

        same, maximum = phase_contract.compare_weights(
            torch.load(native["best_model"], map_location="cpu", weights_only=True),
            torch.load(platform["best_model"], map_location="cpu", weights_only=True),
            parity_atol,
        )
        report["weights_max_abs_diff"] = maximum
        if not same:
            mismatches.append("best_weights")
        if fingerprint != fingerprint_dataset(dataset):
            mismatches.append("dataset_changed")
        report.update(
            status="failed" if mismatches else "passed",
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


def main(*, native=False):
    """Prepare device controls before importing torch, then execute the recipe."""
    import sys

    if native and "--make-initial" in sys.argv:
        parser = argparse.ArgumentParser(
            description="Create shared initial VOC weights"
        )
        parser.add_argument("--make-initial", type=Path, required=True)
        parser.add_argument("--seed", type=int, default=37)
        args = parser.parse_args()
        phase_contract.prepare_process("cpu")
        from .native_detection import make_initial_weights

        make_initial_weights(args.make_initial, args.seed)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "initial_weights", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--concurrent-workload", default="undeclared")
    phase_contract.device_arguments(parser)
    args = parser.parse_args()
    phase_contract.prepare_cli(args, parser)
    runner = run_pair
    if native:
        from .native_detection import run_native

        runner = run_native
    report = runner(**vars(args))
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(
                    args.output
                    / ("native_result.json" if native else "paired_result.json")
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
