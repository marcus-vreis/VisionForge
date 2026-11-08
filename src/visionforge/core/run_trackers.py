"""The contract every optional experiment tracker of a training follows.

MLflow (ADR-170) and Weights & Biases (ADR-172) are built per run folder, opened
when it exists, fed each epoch and closed on every exit path. None of the calls
may raise: a tracker that fails records it beside the run and the training goes
on.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol


class RunTracker(Protocol):
    """One external copy of a local run's parameters and epoch metrics."""

    def start(self, config: dict[str, Any]) -> None:
        """Open the external run with the run's config."""

    def epoch(self, epoch: int, metrics: dict[str, Any]) -> None:
        """Record one finished epoch."""

    def finish(self, status: str) -> None:
        """Close the run as ``FINISHED``, ``KILLED`` or ``FAILED``."""


# Built once the trainer knows the run folder.
TrackerFactory = Callable[[Path], RunTracker]
