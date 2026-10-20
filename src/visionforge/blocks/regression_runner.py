"""TaskRunner adapter for image regression (ADR-041 / ADR-044)."""

from __future__ import annotations

import time
from typing import Any

from visionforge.blocks.regression import RegressionBlock
from visionforge.core.cancellation import CancellationToken
from visionforge.core.task_runner import (
    ProgressCallback,
    RunResult,
    best_epoch_row,
    run_dir_from,
)
from visionforge.utils.regression_config import RegressionConfig

# Regression's test metrics; r2 (higher is better) is the ranking default.
_METRIC_KEYS = ("r2", "rmse", "mae", "mse")


def _validation_metrics(block: RegressionBlock) -> dict[str, float]:
    """The kept checkpoint's validation metrics, under the test metrics' names."""
    train = getattr(block, "_train_result", None)
    best = best_epoch_row(train.history, train.best_epoch) if train else None
    if best is None:
        return {}
    return {k: float(getattr(best, f"val_{k}")) for k in _METRIC_KEYS}


class RegressionRunner:
    """Drives RegressionBlock for one training run, exposing the uniform handle.

    GPU cleanup is the caller's responsibility (the generic comparison runner
    flushes between architectures).
    """

    config_type = RegressionConfig

    # Set by whoever drives the run (the route layer, or the block that owns the
    # comparison) and handed to each block built here, so a stopped job also
    # stops the training in flight (ADR-111). None when nobody can press stop.
    _cancel_token: CancellationToken | None = None
    # Where the block's training events go, set by the orchestrator for each
    # unit it runs (core.task_runner.UnitProgress); None streams nothing.
    _progress_callback: ProgressCallback | None = None

    def run(self, cfg: Any) -> RunResult:
        """Run a single regression trial and return a uniform RunResult."""
        block = RegressionBlock()
        try:
            block.setup(cfg)
            block._cancel_token = self._cancel_token
            block._progress_callback = self._progress_callback
            t0 = time.monotonic()
            block.run()
            elapsed = time.monotonic() - t0

            report = block.report()
            test: dict[str, Any] = report.get("test", {})
            metrics = {
                k: float(test[k]) for k in _METRIC_KEYS if test.get(k) is not None
            }
            selection = _validation_metrics(block)
            return RunResult(
                metrics=metrics,
                status="success",
                training_time_s=elapsed,
                error="",
                stopped=bool(report.get("train", {}).get("stopped")),
                run_dir=run_dir_from(report.get("train", {})),
                selection_metrics=selection,
            )
        except Exception as exc:  # noqa: BLE001
            return RunResult(
                metrics={}, status="failed", training_time_s=None, error=str(exc)
            )

    def metrics(self, result: RunResult) -> dict[str, float]:
        """Return result.metrics unchanged."""
        return result.metrics

    def primary_metric(self) -> str:
        """Return 'r2' — the regression default ranking metric (higher is better)."""
        return "r2"


__all__ = ["RegressionRunner"]
