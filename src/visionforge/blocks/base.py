from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from visionforge.core.cancellation import CancellationToken
from visionforge.utils.config import ExperimentConfig


class ExperimentBlock(ABC):
    """Base class for all experiment strategies."""

    # Set by the GUI layer when a queued job starts, so "stop this run" reaches
    # the trainer that is actually looping (ADR-088). Declared here rather than
    # in each block because the route layer assigns it without knowing which
    # block it built; None is the CLI case, where nobody can press a button.
    _cancel_token: CancellationToken | None = None

    # Set the same way when the researcher asks to continue a stopped run
    # (ADR-092/093): the trainer writes into this directory instead of a new
    # one. None is a fresh run, which is every run started normally.
    _resume_dir: Path | None = None

    @abstractmethod
    def setup(self, config: ExperimentConfig) -> None:
        """Receive the experiment config and prepare internal state."""

    @abstractmethod
    def run(self) -> None:
        """Execute the experiment strategy."""

    @abstractmethod
    def report(self) -> dict[str, Any]:
        """Return a dict of results for logging and GUI display."""


__all__ = ["ExperimentBlock"]
