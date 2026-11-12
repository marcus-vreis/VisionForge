"""What a run may show an experiment tracker: public parameters and metrics.

MLflow, the Hugging Face Hub and Weights & Biases receive the same projection,
so a key that is private for one is private for all (ADR-170, ADR-171, ADR-172).
"""

from __future__ import annotations

import math
import re
from typing import Any

Scalar = str | bool | int | float | None

# Keys never exported: secrets, and anything holding a local path (*_dir too,
# e.g. data.base_dir carries the user's folder and name).
_PRIVATE_PARAMETER = re.compile(
    r"token|password|secret|credential|path|uri|url|(?:^|_)dirs?$", re.I
)
_PARAMETER_NAME = re.compile(r"[\w-]+", re.ASCII)
_METRIC_NAMES = {
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auc",
    "auroc",
    "auc_roc",
    "loss",
    "mse",
    "rmse",
    "mae",
    "r2",
    "miou",
    "dice",
    "pixel_acc",
    "threshold",
    "image_f1",
    "map50",
    "map50_95",
    "best_val_loss",
    "total_epochs",
    "best_epoch",
    "train_loss",
    "val_loss",
    "train_accuracy",
    "val_accuracy",
    "lr",
    "epoch",
}
_SPLIT_PREFIXES = ("test_", "val_", "best_")
_METRIC_KEY = re.compile(r"\w+", re.ASCII)
# Final metrics of the built-in tasks that only the private trackers show; the
# Hub card keeps to the shorter list (ADR-171, ADR-197).
_TRACKER_ONLY_METRICS = {"best_val_miou", "box_loss"}
# run.json history names -> the chart keys the trainer streams to a tracker.
TRACKER_METRIC_KEYS = {
    "train_loss": "loss/train",
    "val_loss": "loss/val",
    "train_accuracy": "accuracy/train",
    "val_accuracy": "accuracy/val",
    "lr": "lr",
}


def public_parameters(config: dict[str, Any]) -> dict[str, Scalar]:
    """Flatten the config to ``{"a.b": scalar}``, leaving private keys out.

    Raises:
        ValueError: a key that is not ``[A-Za-z0-9_-]``, which could make two
            keys flatten to the same name.
    """
    result: dict[str, Scalar] = {}
    _collect(config, "", result)
    return result


def _collect(data: dict[str, Any], prefix: str, result: dict[str, Scalar]) -> None:
    for name, value in sorted(data.items()):
        if not isinstance(name, str) or _PRIVATE_PARAMETER.search(name):
            continue
        if not _PARAMETER_NAME.fullmatch(name):
            raise ValueError(f"Ambiguous parameter name: {name!r}")
        key = f"{prefix}.{name}" if prefix else name
        if isinstance(value, dict):
            _collect(value, key, result)
        elif value is None or isinstance(value, (str, bool, int, float)):
            result[key] = value


def is_metric_name(key: str) -> bool:
    """A known metric, bare or with a split prefix (``test_accuracy``)."""
    if key in _METRIC_NAMES:
        return True
    return any(
        key.startswith(prefix) and key[len(prefix) :] in _METRIC_NAMES
        for prefix in _SPLIT_PREFIXES
    )


def public_metrics(data: Any) -> dict[str, int | float]:
    """The finite numeric values of known metric names, sorted by name."""
    if not isinstance(data, dict):
        return {}
    return {
        key: value
        for key, value in sorted(data.items())
        if is_metric_name(key) and _is_finite_number(value)
    }


def tracker_metrics(record: Any) -> dict[str, int | float]:
    """The final metrics of a run.json that a tracker may show, by their run.json names.

    A built-in task keeps to the public list plus the few task metrics the Hub
    leaves out. A custom task (``custom:<key>``) names its own metrics in its
    code, so every finite number under a plain ``[A-Za-z0-9_]`` name goes
    (ADR-197).
    """
    data = record.get("metrics") if isinstance(record, dict) else None
    if not isinstance(data, dict):
        return {}
    task = record.get("task")
    custom = isinstance(task, str) and task.startswith("custom:")
    return {
        key: value
        for key, value in sorted(data.items())
        if _is_finite_number(value)
        and (
            is_metric_name(key)
            or key in _TRACKER_ONLY_METRICS
            or (custom and bool(_METRIC_KEY.fullmatch(key)))
        )
    }


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )
