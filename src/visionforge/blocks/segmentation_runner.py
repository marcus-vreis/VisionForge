"""TaskRunner adapter for semantic segmentation (ADR-041 / ADR-044)."""

from __future__ import annotations

import time
from typing import Any

from visionforge.blocks.segmentation import SegmentationBlock
from visionforge.core.cancellation import CancellationToken
from visionforge.core.task_runner import RunResult, run_dir_from
from visionforge.utils.segmentation_config import SegmentationConfig

# Segmentation's test metrics; miou (higher is better) is the ranking default.
_METRIC_KEYS = ("miou", "dice", "pixel_acc")


class SegmentationRunner:
    """Drives SegmentationBlock for one training run, exposing the uniform handle.

    GPU cleanup is the caller's responsibility (the generic comparison runner
    flushes between architectures).
    """

    config_type = SegmentationConfig

    # Set by whoever drives the run (the route layer, or the block that owns the
    # comparison) and handed to each block built here, so a stopped job also
    # stops the training in flight (ADR-111). None when nobody can press stop.
    _cancel_token: CancellationToken | None = None

    def run(self, cfg: Any) -> RunResult:
        """Run a single segmentation trial and return a uniform RunResult."""
        block = SegmentationBlock()
        try:
            block.setup(cfg)
            block._cancel_token = self._cancel_token
            t0 = time.monotonic()
            block.run()
            elapsed = time.monotonic() - t0

            report = block.report()
            test: dict[str, Any] = report.get("test", {})
            metrics = {
                k: float(test[k]) for k in _METRIC_KEYS if test.get(k) is not None
            }
            return RunResult(
                metrics=metrics,
                status="success",
                training_time_s=elapsed,
                error="",
                stopped=bool(report.get("train", {}).get("stopped")),
                run_dir=run_dir_from(report.get("train", {})),
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
