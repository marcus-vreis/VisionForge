"""TaskRunner adapter for anomaly detection (ADR-041 / ADR-044 / ADR-045)."""

from __future__ import annotations

import time
from typing import Any

from visionforge.blocks.anomaly import AnomalyBlock
from visionforge.core.task_runner import RunResult
from visionforge.utils.anomaly_config import AnomalyConfig

# Anomaly's test metrics; auroc (higher is better) is the ranking default.
_METRIC_KEYS = ("auroc", "image_f1")


class AnomalyRunner:
    """Drives AnomalyBlock for one training run, exposing the uniform handle.

    GPU cleanup is the caller's responsibility (the generic comparison/sweep
    runners flush between trials).
    """

    config_type = AnomalyConfig

    def run(self, cfg: Any) -> RunResult:
        """Run a single anomaly trial and return a uniform RunResult."""
        block = AnomalyBlock()
        try:
            block.setup(cfg)
            t0 = time.monotonic()
            block.run()
            elapsed = time.monotonic() - t0

            test: dict[str, Any] = block.report().get("test", {})
            metrics = {
                k: float(test[k]) for k in _METRIC_KEYS if test.get(k) is not None
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
        """Return 'auroc' — the anomaly default ranking metric."""
        return "auroc"


__all__ = ["AnomalyRunner"]
