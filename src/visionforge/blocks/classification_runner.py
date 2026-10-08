from __future__ import annotations

import time
from typing import Any

from visionforge.blocks.classification import ClassificationBlock
from visionforge.core.cancellation import CancellationToken
from visionforge.core.task_runner import RunResult
from visionforge.utils.config import ExperimentConfig


class ClassificationRunner:
    """TaskRunner adapter that drives ClassificationBlock for one training run.

    GPU cleanup (gc.collect / torch.cuda.empty_cache) is the caller's
    responsibility — kept out of this module to keep the call-count assertions
    in ModelComparisonBlock's tests exact.
    """

    config_type = ExperimentConfig

    # Set by whoever drives the run (the route layer, or the block that owns the
    # comparison) and handed to each block built here, so a stopped job also
    # stops the training in flight (ADR-111). None when nobody can press stop.
    _cancel_token: CancellationToken | None = None

    def run(self, cfg: Any) -> RunResult:
        """Run a single classification training trial and return a uniform RunResult."""
        block = ClassificationBlock()
        try:
            block.setup(cfg)
            block._cancel_token = self._cancel_token
            t0 = time.monotonic()
            block.run()
            elapsed = time.monotonic() - t0

            report = block.report()
            eval_metrics: dict[str, Any] = report.get("eval", {})
            metrics: dict[str, float] = {
                k: float(v)
                for k, v in eval_metrics.items()
                if k in ("accuracy", "f1", "auc_roc") and v is not None
            }
            return RunResult(
                metrics=metrics,
                status="success",
                training_time_s=elapsed,
                error="",
                stopped=bool(report.get("train", {}).get("stopped")),
            )
        except Exception as exc:  # noqa: BLE001
            return RunResult(
                metrics={}, status="failed", training_time_s=None, error=str(exc)
            )

    def metrics(self, result: RunResult) -> dict[str, float]:
        """Return result.metrics unchanged."""
        return result.metrics

    def primary_metric(self) -> str:
        """Return 'accuracy' — the classification default ranking metric."""
        return "accuracy"


__all__ = ["ClassificationRunner"]
