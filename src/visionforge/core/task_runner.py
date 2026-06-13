from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class RunResult:
    """Uniform result envelope returned by every TaskRunner implementation."""

    metrics: dict[str, float] = field(default_factory=dict)
    status: str = "failed"
    training_time_s: float | None = None
    error: str = ""


@runtime_checkable
class TaskRunner(Protocol):
    """Protocol for task-agnostic orchestration (comparison, sweep, batch-predict).

    Slice 1 exposes only the comparison-relevant surface. load_checkpoint and
    predict are deferred to later slices.
    """

    config_type: type[Any]
    """The task's Pydantic config class — lets a generic orchestrator validate a
    per-trial config dict without knowing which task it is."""

    def run(self, cfg: Any) -> RunResult:
        """Execute one full training run and return a uniform result."""
        ...

    def metrics(self, result: RunResult) -> dict[str, float]:
        """Extract the metric dict from a RunResult (allows adapter-level rename/filter)."""
        ...

    def primary_metric(self) -> str:
        """Return the task's canonical ranking metric name (e.g. 'accuracy', 'map50')."""
        ...


__all__ = ["RunResult", "TaskRunner"]
