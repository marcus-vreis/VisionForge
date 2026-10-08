from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from visionforge.core.cancellation import CancellationToken


@dataclass
class RunResult:
    """Uniform result envelope returned by every TaskRunner implementation."""

    metrics: dict[str, float] = field(default_factory=dict)
    status: str = "failed"
    training_time_s: float | None = None
    error: str = ""
    # True only when the stop cut this training (ADR-111). The orchestrators
    # mark a unit stopped from this, never from the token alone: a unit whose
    # stop landed in its last epoch, or a custom task that owns its loop, ran
    # to its end and counts.
    stopped: bool = False


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


# The runner carries the job's stop signal the way a block carries
# `_cancel_token` (ADR-094): it hands the token to every block it builds, so the
# training in flight stops at its epoch boundary, and the orchestrator driving it
# reads the same token between units, so nothing new starts (ADR-111). An
# attribute rather than a `run()` argument, so the protocol every test double
# implements stays as it is.
def give_cancel_token(runner: Any, token: CancellationToken | None) -> None:
    """Hand ``runner`` the job's stop signal, if it declares a slot for one."""
    if hasattr(runner, "_cancel_token"):
        runner._cancel_token = token


def runner_cancel_token(runner: object) -> CancellationToken | None:
    """The stop signal ``runner`` carries, or None when it has none."""
    token = getattr(runner, "_cancel_token", None)
    return token if isinstance(token, CancellationToken) else None


__all__ = ["RunResult", "TaskRunner", "give_cancel_token", "runner_cancel_token"]
