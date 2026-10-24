"""TaskRunner adapter for researcher-defined tasks (ADR-058).

Wraps :class:`~visionforge.tasks.engine.GenericTaskEngine` behind the uniform
``TaskRunner`` handle (ADR-041), so a custom task gets hyperparameter sweeps
(ADR-045/052) and multi-seed replicates (ADR-056) for free — the same
orchestrators the built-in tasks use, no task-specific code.

Model comparison is deliberately not exposed for custom tasks: it overrides
``model.name``, which ``BaseTaskConfig`` does not guarantee — and comparing
alternatives is just a one-axis sweep over whichever field the task declares.
"""

from __future__ import annotations

import time
from typing import Any

from visionforge.core.cancellation import CancellationToken
from visionforge.core.task_runner import RunResult, best_epoch_row
from visionforge.tasks.base import TaskSpec
from visionforge.tasks.engine import GenericTaskEngine
from visionforge.tasks.registry import TaskInfo
from visionforge.utils.messages import tr


class CustomTaskRunner:
    """Drives one custom-task training per trial for the generic orchestrators."""

    # Handed to the engine each trial, the way the built-in runners hand it to
    # their blocks (ADR-111). A Level 2 task owns its loop and never reads it;
    # the orchestrator still stops between trials.
    _cancel_token: CancellationToken | None = None

    def __init__(self, info: TaskInfo) -> None:
        if info.spec_cls is None:
            raise ValueError(f"Task '{info.key}' has no spec class registered.")
        self._info = info
        self.config_type = info.spec_cls.Config
        # Read by the orchestrators to rank trials the way the task declared
        # (`runner_metric_direction`): "score" may well be lower-is-better.
        self.metric_directions = dict(info.metrics)

    @property
    def selection_unavailable(self) -> str | None:
        """Explain why Level 2 selection is unavailable in the caller's language."""
        # A Level 2 `run` returns metrics of a split VisionForge cannot see, so
        # a sweep could be ranking on test numbers (ADR-131). Level 1 scores
        # every epoch on its val loader and selects on that.
        return (
            tr("sweep.level2_selection_unavailable", task=self._info.key)
            if getattr(self._info.spec_cls, "run", TaskSpec.run) is not TaskSpec.run
            else None
        )

    def run(self, cfg: Any) -> RunResult:
        """Run a single training trial and return a uniform RunResult."""
        try:
            t0 = time.monotonic()
            result = GenericTaskEngine(self._info, cfg).run(
                cancel_token=self._cancel_token
            )
            best = best_epoch_row(
                getattr(result, "history", []), getattr(result, "best_epoch", 0)
            )
            return RunResult(
                metrics=dict(result.metrics),
                # The kept epoch's scores on the val loader, never the
                # `test_`-prefixed final pass.
                selection_metrics=dict(best.val_metrics) if best else {},
                status="success",
                training_time_s=time.monotonic() - t0,
                error="",
                stopped=result.stopped,
                # The orchestrators tag the seed's run with its group and
                # drop a cut unit's resume state through this (ADR-111/113);
                # without it a custom task's group finds no seed.
                run_dir=result.run_dir,
            )
        except Exception as exc:  # noqa: BLE001 — a failed trial must not abort the sweep
            return RunResult(
                metrics={}, status="failed", training_time_s=None, error=str(exc)
            )

    def metrics(self, result: RunResult) -> dict[str, float]:
        """Return result.metrics unchanged."""
        return result.metrics

    def primary_metric(self) -> str:
        """The metric declared as primary in ``@register_task``."""
        return self._info.primary_metric


__all__ = ["CustomTaskRunner"]
