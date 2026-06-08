from __future__ import annotations

import time
from typing import Any

from visionforge.blocks.regression import RegressionBlock
from visionforge.core.task_runner import RunResult
from visionforge.utils.regression_config import RegressionConfig


class RegressionRunner:
    """TaskRunner adapter that drives RegressionBlock for one training run.

    GPU cleanup (gc.collect / torch.cuda.empty_cache) is the caller's
    responsibility — kept out of this module so the adapter stays free of torch,
    mirroring ClassificationRunner (ADR-041).
    """

    config_type = RegressionConfig

    def run(self, cfg: Any) -> RunResult:
        """Run a single regression training trial and return a uniform RunResult."""
        block = RegressionBlock()
        try:
            block.setup(cfg)
            t0 = time.monotonic()
            block.run()
            elapsed = time.monotonic() - t0

            report = block.report()
            test_metrics: dict[str, Any] = report.get("test", {})
            metrics: dict[str, float] = {
                k: float(v)
                for k, v in test_metrics.items()
                if k in ("mse", "rmse", "mae", "r2") and v is not None
            }
            return RunResult(
                metrics=metrics, status="success", training_time_s=elapsed, error=""
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
