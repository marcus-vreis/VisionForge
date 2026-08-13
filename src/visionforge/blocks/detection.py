"""Object-detection experiment block (Ultralytics).

Mirrors the `ExperimentBlock` setup/run/report contract but over the standalone
`DetectionConfig` tree rather than `ExperimentConfig` (see ADR-033). It is not an
`ExperimentBlock` subclass — that ABC is bound to the classification config — so
it is dispatched directly by the detection run endpoint, not the block registry.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from visionforge.core.cancellation import CancellationToken
from visionforge.core.detection_trainer import DetectionTrainer, DetectionTrainResult
from visionforge.utils.detection_config import DetectionConfig


class DetectionBlock:
    """End-to-end detection training block over `DetectionConfig`."""

    def setup(self, config: DetectionConfig) -> None:
        self._config = config
        self._result: DetectionTrainResult | None = None
        # Injected by the GUI layer to stream live epoch progress via SSE.
        self._progress_callback: Callable[[dict[str, Any]], None] | None = None
        # Set by the GUI layer when a queued job starts, so "stop this run"
        # reaches the trainer that is looping (ADR-088). None is the CLI case.
        self._cancel_token: CancellationToken | None = None

    def run(self) -> None:
        self._result = DetectionTrainer(self._config).fit(
            progress_callback=self._progress_callback,
            cancel_token=self._cancel_token,
        )

    def report(self) -> dict[str, Any]:
        """Return a summary of the run for logging and GUI display."""
        if self._result is None:
            return {}
        r = self._result
        best_map50 = next((h.map50 for h in r.history if h.epoch == r.best_epoch), None)
        return {
            "detection": {
                "best_epoch": r.best_epoch,
                "best_map50_95": r.best_map50_95,
                "best_map50": best_map50,
                "total_epochs": r.total_epochs,
                "device_used": r.device_used,
                "model_path": str(r.model_path),
                "run_dir": str(r.run_dir),
            }
        }


__all__ = ["DetectionBlock"]
