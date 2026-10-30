"""Per-trial experiment names for sweeps.

A sweep used to run every trial under the experiment's own name, so History
listed ten runs called ``cifar10_resnet18_e20`` and the value that set them
apart lived only in the trial table. Each trial now carries what it varies:
``cifar10_resnet18_e20_lr0.002``. The short keys match the names the GUI
suggests for a new experiment (``frontend/src/lib/experiment-name.ts``).
"""

from __future__ import annotations

import re
from typing import Any

# Last segment of the dotted path -> the short form used in names.
SHORT_KEYS: dict[str, str] = {
    "learning_rate": "lr",
    "lr0": "lr",
    "batch_size": "bs",
    "weight_decay": "wd",
    "epochs": "e",
    "momentum": "mom",
    "dropout": "do",
    "image_size": "img",
    "imgsz": "img",
    "optimizer": "opt",
    "loss": "loss",
}

_UNSAFE = re.compile(r"[^A-Za-z0-9_.+-]+")


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, float):
        return format(value, "g")
    return _UNSAFE.sub("-", str(value)).strip("-")


def _short_key(dot_key: str) -> str:
    last = dot_key.rsplit(".", 1)[-1]
    if dot_key == "model.name":
        return ""
    return SHORT_KEYS.get(last, last)


def trial_name(base: str, overrides: dict[str, Any]) -> str:
    """Return ``base`` followed by one ``<key><value>`` part per override.

    ``model.name`` contributes only its value (the architecture already reads
    as a name). Empty ``overrides`` returns ``base`` unchanged.
    """
    parts = [f"{_short_key(k)}{_format_value(v)}" for k, v in overrides.items()]
    parts = [p for p in parts if p]
    return "_".join([base, *parts]) if parts else base
