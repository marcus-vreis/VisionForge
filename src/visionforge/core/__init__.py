"""Core training, evaluation and plotting API.

The public names are resolved on first access (PEP 562), so importing a light
submodule such as ``visionforge.core.cancellation`` does not pull torch,
scikit-learn or matplotlib into the process (GUI-11, #202).
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from visionforge.core.data import DataModule
    from visionforge.core.evaluator import EvalResult, Evaluator
    from visionforge.core.plotter import MetricsPlotter
    from visionforge.core.trainer import EpochResult, Trainer, TrainResult

_EVALUATOR = "visionforge.core.evaluator"
_TRAINER = "visionforge.core.trainer"

_LAZY: dict[str, str] = {
    "DataModule": "visionforge.core.data",
    "Evaluator": _EVALUATOR,
    "EvalResult": _EVALUATOR,
    "MetricsPlotter": "visionforge.core.plotter",
    "Trainer": _TRAINER,
    "TrainResult": _TRAINER,
    "EpochResult": _TRAINER,
}

__all__ = [
    "DataModule",
    "Evaluator",
    "EvalResult",
    "MetricsPlotter",
    "Trainer",
    "TrainResult",
    "EpochResult",
]


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
