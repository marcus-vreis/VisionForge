"""Run the seed-variance study through VisionForge's own replicated comparison.

Every (dataset, architecture) pair trains the three pre-registered variants on
the same ten seeds; one report per pair lands in ``results/``. A pair whose
report exists is skipped, so an interrupted study resumes where it stopped.

Usage:
    python research/seed_variance/run_study.py --check   # validate configs, train nothing
    python research/seed_variance/run_study.py --pilot   # timing + determinism check
    python research/seed_variance/run_study.py           # the full grid (PROTOCOL.md)
    python research/seed_variance/run_study.py --only usk_full:resnet18
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from visionforge import __version__
from visionforge.blocks.classification_runner import ClassificationRunner
from visionforge.core.learning_rate import suggested_learning_rate, suggested_optimizer
from visionforge.core.replicated_comparison import run_replicated_comparison
from visionforge.core.replicates import run_replicates
from visionforge.utils.config import ExperimentConfig

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / "results"

# Pre-registered in PROTOCOL.md; changing any of these is a deviation to log there.
DATASETS: dict[str, dict[str, Any]] = {
    "usk_full": {
        "base_dir": ROOT / "datasets" / "Classification - USK-COFFEE",
        "num_classes": 4,
        "val_dir": "val",
    },
    "usk_10pct": {
        "base_dir": ROOT / "datasets" / "Classification - USK-COFFEE-10pct",
        "num_classes": 4,
        "val_dir": "val",
    },
    "beans": {
        "base_dir": ROOT / "datasets" / "Classification - beans",
        "num_classes": 3,
        "val_dir": "validation",
    },
}
# Two CNNs and a transformer (PROTOCOL.md): whether the seeds a comparison needs
# differ between the families.
ARCHITECTURES = ["resnet18", "convnext_tiny", "swin_t"]
VARIANTS: dict[str, dict[str, Any]] = {
    "baseline": {},
    "no_aug": {
        "data.transforms.horizontal_flip": False,
        "data.transforms.rotation_degrees": 0,
    },
    "res160": {"data.transforms.image_size": 160},
}
SEEDS = list(range(10))
METRIC = "accuracy"


def base_config(dataset: str, arch: str) -> dict[str, Any]:
    """The baseline config for one pair; variants override it by dot-path."""
    spec = DATASETS[dataset]
    optimizer = suggested_optimizer(arch)
    return {
        "name": f"seedvar_{dataset}_{arch}",
        "task": "multiclass",
        "block": "classification",
        "model": {"name": arch, "num_classes": spec["num_classes"], "pretrained": True},
        "training": {
            "epochs": 20,
            "early_stopping_patience": 5,
            "batch_size": 32,
            "optimizer": optimizer,
            "learning_rate": suggested_learning_rate(arch, optimizer),
            "weight_decay": 0.0,
            "seed": 0,
            "deterministic": True,
        },
        "data": {
            "base_dir": str(spec["base_dir"]),
            "train_dir": "train",
            "val_dir": spec["val_dir"],
            "test_dir": "test",
            "num_workers": -1,
            "transforms": {
                "image_size": 224,
                "horizontal_flip": True,
                "rotation_degrees": 10,
                "color_jitter": False,
            },
        },
        "device": {"kind": "cuda"},
    }


def pairs(only: str | None) -> list[tuple[str, str]]:
    everything = [(d, a) for d in DATASETS for a in ARCHITECTURES]
    if only is None:
        return everything
    dataset, arch = only.split(":")
    if (dataset, arch) not in everything:
        raise SystemExit(f"Unknown pair {only!r}.")
    return [(dataset, arch)]


def check() -> int:
    """Validate every config and variant; report datasets not on disk yet."""
    missing = [d for d, spec in DATASETS.items() if not spec["base_dir"].is_dir()]
    for dataset in missing:
        print(f"[missing] {dataset}: {DATASETS[dataset]['base_dir']}")
    for dataset, arch in pairs(None):
        if dataset in missing:
            continue
        for overrides in VARIANTS.values():
            cfg = base_config(dataset, arch)
            for path, value in overrides.items():
                node = cfg
                *parents, leaf = path.split(".")
                for key in parents:
                    node = node[key]
                node[leaf] = value
            ExperimentConfig.model_validate(cfg)
        print(f"[ok] {dataset}:{arch}  ({len(VARIANTS)} variants x {len(SEEDS)} seeds)")
    return len(missing)


def pilot() -> None:
    """Seed 0 twice for resnet18 (must match: deterministic), once each for the others.

    Writes ``results/pilot.json`` with the wall time of each run, which is what
    the full grid's duration is estimated from.
    """
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    runner = ClassificationRunner()
    for dataset in DATASETS:
        for arch, repeats in (("resnet18", 2), ("convnext_tiny", 1), ("swin_t", 1)):
            for _ in range(repeats):
                started = time.time()
                (trial,) = run_replicates(
                    runner, base_config(dataset, arch), [0], METRIC
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "architecture": arch,
                        "status": trial.status,
                        "metric": trial.metrics.get(METRIC),
                        "wall_time_s": round(time.time() - started, 1),
                        "error": trial.error,
                    }
                )
                print(rows[-1])
    (RESULTS / "pilot.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def run_pair(dataset: str, arch: str) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{dataset}__{arch}.json"
    if out.exists():
        print(f"[skip] {out.name} exists")
        return out
    started = time.time()
    report = run_replicated_comparison(
        ClassificationRunner(),
        base_config(dataset, arch),
        VARIANTS,
        SEEDS,
        METRIC,
        direction="higher",
    )
    report["study"] = {
        "dataset": dataset,
        "architecture": arch,
        "wall_time_s": round(time.time() - started, 1),
        "visionforge_version": __version__,
    }
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[done] {out.name} in {report['study']['wall_time_s']} s")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--pilot", action="store_true")
    parser.add_argument("--only", help="dataset:architecture, e.g. usk_full:resnet18")
    args = parser.parse_args()

    if args.check:
        raise SystemExit(check())
    if args.pilot:
        pilot()
        return
    for dataset, arch in pairs(args.only):
        run_pair(dataset, arch)


if __name__ == "__main__":
    main()
