"""TaskRunner adapter for object detection (ADR-041 / ADR-044 / ADR-045)."""

from __future__ import annotations

import time
from typing import Any

from visionforge.blocks.detection import DetectionBlock
from visionforge.core.task_runner import RunResult
from visionforge.utils.detection_config import DetectionConfig

# Detection's ranking metrics. map50_95 (Ultralytics) is the default; map50 is
# also exposed so torchvision runs (which have no map50_95) can still be ranked.
_METRIC_KEYS = {"map50_95": "best_map50_95", "map50": "best_map50"}


class DetectionRunner:
    """Drives DetectionBlock for one training run, exposing the uniform handle.

    GPU cleanup is the caller's responsibility (the generic comparison/sweep
    runners flush between trials).
    """

    config_type = DetectionConfig

    def run(self, cfg: Any) -> RunResult:
        """Run a single detection trial and return a uniform RunResult."""
        block = DetectionBlock()
        try:
            block.setup(cfg)
            t0 = time.monotonic()
            block.run()
            elapsed = time.monotonic() - t0

            det: dict[str, Any] = block.report().get("detection", {})
            metrics = {
                name: float(det[src])
                for name, src in _METRIC_KEYS.items()
                if det.get(src) is not None
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
        """Return 'map50_95' — the detection default ranking metric."""
        return "map50_95"


__all__ = ["DetectionRunner"]
