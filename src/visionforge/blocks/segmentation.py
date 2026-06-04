"""Semantic-segmentation experiment block (Phase 8).

Mirrors the `ExperimentBlock` setup/run/report contract but over the standalone
`SegmentationConfig` tree rather than `ExperimentConfig` (see ADR-037). It is not
an `ExperimentBlock` subclass — that ABC is bound to the classification config —
so it is dispatched directly by the segmentation run endpoint, not the block
registry. Same rationale as detection's ADR-033 and regression's ADR-036.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from visionforge.core.plotter import MetricsPlotter
from visionforge.core.segmentation_data import SegmentationDataModule
from visionforge.core.segmentation_trainer import (
    SegmentationTrainer,
    SegmentationTrainResult,
)
from visionforge.models.segmentation_factory import SegmentationModelFactory
from visionforge.utils.segmentation_config import SegmentationConfig


class SegmentationBlock:
    """End-to-end semantic-segmentation training block over `SegmentationConfig`."""

    def setup(self, config: SegmentationConfig) -> None:
        self._config = config
        self._train_result: SegmentationTrainResult | None = None
        self._test_metrics: tuple[float, float, float] | None = None
        # Injected by the GUI layer to stream live epoch progress via SSE.
        self._progress_callback: Callable[[dict[str, Any]], None] | None = None

    def run(self) -> None:
        model = SegmentationModelFactory.create(self._config.model)
        data = SegmentationDataModule(self._config)
        trainer = SegmentationTrainer(self._config)

        self._train_result = trainer.fit(
            model, data, progress_callback=self._progress_callback
        )

        # Reload the best checkpoint before test-set evaluation.
        state_dict = torch.load(
            str(self._train_result.model_path), map_location="cpu", weights_only=True
        )
        model.load_state_dict(state_dict)  # type: ignore[arg-type]

        test_loader = data.test_loader()
        if test_loader is not None:
            self._test_metrics = trainer.evaluate(model, test_loader)

        run_dir = self._train_result.model_path.parent
        graphics = self._render_plots(run_dir)
        self._update_run_json(run_dir, graphics)

    def report(self) -> dict[str, Any]:
        """Return a summary of the run for logging and GUI display."""
        result: dict[str, Any] = {}
        if self._train_result is not None:
            r = self._train_result
            result["train"] = {
                "best_epoch": r.best_epoch,
                "best_val_miou": r.best_val_miou,
                "total_epochs": r.total_epochs,
                "device_used": r.device_used,
                "run_dir": str(r.model_path.parent),
            }
        if self._test_metrics is not None:
            miou, dice, pixel_acc = self._test_metrics
            result["test"] = {"miou": miou, "dice": dice, "pixel_acc": pixel_acc}
        return result

    # ── private ───────────────────────────────────────────────────────────────

    def _render_plots(self, run_dir: Path) -> list[Path]:
        """Render the train/val loss curve (reuses the classification plotter)."""
        assert self._train_result is not None
        loss_path = run_dir / "loss.png"
        # SegmentationEpochResult exposes epoch/train_loss/val_loss, which is all
        # loss_curve reads — duck-typed reuse of the classification plotter.
        MetricsPlotter.loss_curve(self._train_result.history, loss_path)  # type: ignore[arg-type]
        return [loss_path]

    def _update_run_json(self, run_dir: Path, graphics: list[Path]) -> None:
        """Rewrite run.json with test metrics and artifact paths."""
        run_json_path = run_dir / "run.json"
        if not run_json_path.exists():
            return

        data: dict[str, Any] = json.loads(run_json_path.read_text(encoding="utf-8"))
        if self._test_metrics is not None:
            miou, dice, pixel_acc = self._test_metrics
            data["metrics"].update(
                {
                    "test_miou": miou,
                    "test_dice": dice,
                    "test_pixel_acc": pixel_acc,
                }
            )
        data["artifacts"]["graphics"] = [str(p) for p in graphics]
        run_json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


__all__ = ["SegmentationBlock"]
