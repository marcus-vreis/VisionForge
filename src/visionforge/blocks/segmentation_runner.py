from __future__ import annotations

import time
from typing import Any

from visionforge.blocks.segmentation import SegmentationBlock
from visionforge.core.task_runner import RunResult
from visionforge.utils.segmentation_config import SegmentationConfig


class SegmentationRunner:
    """TaskRunner adapter that drives SegmentationBlock for one training run.

    GPU cleanup (gc.collect / torch.cuda.empty_cache) is the caller's
    responsibility — kept out of this module so the adapter stays free of torch,
    mirroring ClassificationRunner (ADR-041).
    """

    config_type = SegmentationConfig

    def run(self, cfg: Any) -> RunResult:
        """Run a single segmentation training trial and return a uniform RunResult."""
        block = SegmentationBlock()
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
                if k in ("miou", "dice", "pixel_acc") and v is not None
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
