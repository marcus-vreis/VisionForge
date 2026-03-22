from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from visionforge.utils.config import ExperimentConfig


class ExperimentBlock(ABC):
    """Base class for all experiment strategies."""

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
