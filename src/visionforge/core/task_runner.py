from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from visionforge.core.cancellation import CancellationToken
from visionforge.core.resume import discard_resume_state
from visionforge.core.significance import infer_direction


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
    # The directory the unit trained into, when it has one. The orchestrator
    # needs it to drop the resume state of a unit the stop cut.
    run_dir: Path | None = None


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


def run_dir_from(section: dict[str, Any]) -> Path | None:
    """The run directory a block's report section names, if it names one."""
    raw = section.get("run_dir")
    return Path(raw) if raw else None


def discard_cut_unit_resume(result: RunResult) -> None:
    """Make a unit the stop cut non-resumable, since its job never finished.

    A trainer keeps its resume state when the token cuts it, which is what lets
    a stopped single run be continued. Inside a sweep, comparison or replicate
    set the same unit is one cell of a table that stays `stopped`; continuing it
    alone would give History a run detached from its job (ADR-111).
    """
    if result.run_dir is not None:
        discard_resume_state(result.run_dir)


def runner_metric_direction(runner: object, metric: str) -> Literal["higher", "lower"]:
    """Whether ``metric`` is better high or low for this runner's task.

    A task that declares its metrics (``@register_task(metrics=...)``, carried
    as ``metric_directions``) is believed; otherwise the name decides, by the
    same ``infer_direction`` rule the replicated comparison uses.
    """
    declared = getattr(runner, "metric_directions", None)
    if isinstance(declared, dict):
        direction = declared.get(metric)
        if direction == "higher":
            return "higher"
        if direction == "lower":
            return "lower"
    return infer_direction(metric)


def rank_by_metric[T](
    items: list[T],
    value: Callable[[T], float | None],
    direction: str,
) -> list[T]:
    """Best first by ``value``; items without a value go last, in their order.

    Sorting descending with a 0.0 stand-in for a missing value used to put the
    worst trial first for a lower-is-better metric, and a trial that reported
    nothing above every negative R².
    """

    def key(item: T) -> tuple[int, float]:
        number = value(item)
        if number is None:
            return (1, 0.0)
        return (0, -number if direction == "higher" else number)

    return sorted(items, key=key)


__all__ = [
    "RunResult",
    "TaskRunner",
    "discard_cut_unit_resume",
    "give_cancel_token",
    "rank_by_metric",
    "run_dir_from",
    "runner_cancel_token",
    "runner_metric_direction",
]
