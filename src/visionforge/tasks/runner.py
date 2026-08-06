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

from visionforge.core.task_runner import RunResult
from visionforge.tasks.engine import GenericTaskEngine
from visionforge.tasks.registry import TaskInfo


class CustomTaskRunner:
    """Drives one custom-task training per trial for the generic orchestrators."""

    def __init__(self, info: TaskInfo) -> None:
        if info.spec_cls is None:
            raise ValueError(f"Task '{info.key}' has no spec class registered.")
        self._info = info
        self.config_type = info.spec_cls.Config

    def run(self, cfg: Any) -> RunResult:
        """Run a single training trial and return a uniform RunResult."""
        try:
            t0 = time.monotonic()
            result = GenericTaskEngine(self._info, cfg).run()
            return RunResult(
                metrics=dict(result.metrics),
                status="success",
                training_time_s=time.monotonic() - t0,
                error="",
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
