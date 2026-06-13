"""TaskRunner adapter for semantic segmentation (ADR-041 / ADR-044)."""

from __future__ import annotations

import time
from typing import Any

from visionforge.blocks.segmentation import SegmentationBlock
from visionforge.core.task_runner import RunResult
from visionforge.utils.segmentation_config import SegmentationConfig

# Segmentation's test metrics; miou (higher is better) is the ranking default.
_METRIC_KEYS = ("miou", "dice", "pixel_acc")


class SegmentationRunner:
    """Drives SegmentationBlock for one training run, exposing the uniform handle.

    GPU cleanup is the caller's responsibility (the generic comparison runner
    flushes between architectures).
    """

    config_type = SegmentationConfig

    def run(self, cfg: Any) -> RunResult:
        """Run a single segmentation trial and return a uniform RunResult."""
        block = SegmentationBlock()
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
        """Return 'miou' — the segmentation default ranking metric (higher is better)."""
        return "miou"


__all__ = ["SegmentationRunner"]
