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
    direction: Literal["higher", "lower"],
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


ProgressCallback = Callable[[dict[str, Any]], None]


def trial_progress_wrapper(
    progress_callback: ProgressCallback | None,
    trial_idx: int,
    total_trials: int,
) -> ProgressCallback | None:
    """Wrap a job-level callback so one unit's trainer events carry its context.

    The GUI closes the SSE stream on a bare 'end'; since each unit's trainer emits
    its own 'end', the inner one is rewritten to 'trial_end' to keep the stream
    open across units. The inner 'start' is dropped because the job emits its own
    'trial_start'. Returns None when there is nothing to forward.
    """
    if progress_callback is None:
        return None

    def _wrapped(event: dict[str, Any]) -> None:
        kind = event.get("event")
        if kind == "start":
            return
        annotated = {**event, "trial_index": trial_idx, "total_trials": total_trials}
        if kind == "end":
            annotated["event"] = "trial_end"
        progress_callback(annotated)

    return _wrapped


def give_progress_callback(runner: Any, callback: ProgressCallback | None) -> None:
    """Hand ``runner`` the events sink of its next run, if it declares a slot."""
    if hasattr(runner, "_progress_callback"):
        runner._progress_callback = callback


class UnitProgress:
    """The live stream of a job that trains one unit after another through a runner.

    A model comparison trains one architecture per unit, a sweep one point of its
    grid. Without this their streams carried no epochs, and the training sheet sat
    on "preparing" for every unit (ADR-118). Each unit is announced with
    ``trial_start`` (what it varies in ``overrides``), its trainer's epochs are
    forwarded with the unit's index, and it is closed with exactly one
    ``trial_end`` that carries the unit's outcome (``status``) and, when the
    trainer reported its end, how many epochs it ran (``total_epochs``). The
    trainer's own ``end`` is held back for that, so it never reaches the stream as
    a terminal ``end`` and a unit that died before its trainer spoke is closed too.
    """

    def __init__(self, callback: ProgressCallback | None, total: int) -> None:
        self._callback = callback
        self._total = total
        self._open = False
        self._trainer_end: dict[str, Any] | None = None

    def start(self, runner: Any, index: int, overrides: dict[str, Any]) -> None:
        """Announce unit ``index`` and route ``runner``'s events into the stream."""
        if self._callback is None:
            return
        callback = self._callback
        callback(
            {
                "event": "trial_start",
                "trial_index": index,
                "total_trials": self._total,
                "overrides": overrides,
            }
        )
        self._open = True
        self._trainer_end = None

        def forward(event: dict[str, Any]) -> None:
            kind = event.get("event")
            if kind == "start":
                return  # the unit was announced already
            if kind == "end":
                self._trainer_end = event
                return
            callback({**event, "trial_index": index, "total_trials": self._total})

        give_progress_callback(runner, forward)

    def finish(self, runner: Any, index: int, status: str) -> None:
        """Close unit ``index`` with its outcome, and detach ``runner``."""
        if self._callback is None or not self._open:
            return
        give_progress_callback(runner, None)
        event: dict[str, Any] = {
            "event": "trial_end",
            "trial_index": index,
            "total_trials": self._total,
            "status": status,
        }
        if self._trainer_end is not None and "total_epochs" in self._trainer_end:
            event["total_epochs"] = self._trainer_end["total_epochs"]
        self._callback(event)
        self._open = False

    def end(self, ran: int) -> None:
        """The job's single terminal ``end``, after its last unit."""
        if self._callback is not None:
            self._callback({"event": "end", "total_epochs": 0, "total_trials": ran})


__all__ = [
    "ProgressCallback",
    "RunResult",
    "TaskRunner",
    "UnitProgress",
    "discard_cut_unit_resume",
    "give_cancel_token",
    "give_progress_callback",
    "rank_by_metric",
    "run_dir_from",
    "runner_cancel_token",
    "runner_metric_direction",
    "trial_progress_wrapper",
]
