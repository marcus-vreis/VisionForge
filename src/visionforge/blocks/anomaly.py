"""Anomaly-detection experiment block (Phase 9).

Mirrors the `ExperimentBlock` setup/run/report contract but over the standalone
`AnomalyConfig` tree rather than `ExperimentConfig` (see ADR-038). It is not an
`ExperimentBlock` subclass — that ABC is bound to the classification config — so
it is dispatched directly by the anomaly run endpoint, not the block registry.
Same rationale as ADR-033 (detection) / ADR-036 (regression) / ADR-037 (segmentation).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from visionforge.core.anomaly_data import AnomalyDataModule
from visionforge.core.anomaly_trainer import AnomalyTrainer, AnomalyTrainResult
from visionforge.core.metric_ci import MetricCI, bootstrap_anomaly_cis
from visionforge.core.plotter import MetricsPlotter
from visionforge.models.anomaly_factory import AnomalyModelFactory
from visionforge.utils.anomaly_config import AnomalyConfig


class AnomalyBlock:
    """End-to-end anomaly-detection training block over `AnomalyConfig`."""

    def setup(self, config: AnomalyConfig) -> None:
        self._config = config
        self._train_result: AnomalyTrainResult | None = None
        self._test_metrics: tuple[float, float, float] | None = None
        self._metric_cis: dict[str, MetricCI] = {}
        # Injected by the GUI layer to stream live progress via SSE.
        self._progress_callback: Callable[[dict[str, Any]], None] | None = None

    def run(self) -> None:
        model = AnomalyModelFactory.create(self._config.model)
        data = AnomalyDataModule(self._config)
        trainer = AnomalyTrainer(self._config)

        self._train_result = trainer.fit(
            model, data, progress_callback=self._progress_callback
        )

        # Reload the best checkpoint before final scoring.
        state_dict = torch.load(
            str(self._train_result.model_path), map_location="cpu", weights_only=True
        )
        model.load_state_dict(state_dict)  # type: ignore[arg-type]

        self._test_metrics, labels, scores = trainer.evaluate_with_scores(
            model, data.train_loader(), data.test_loader()
        )
        self._metric_cis = bootstrap_anomaly_cis(
            labels,
            scores,
            self._test_metrics[1],  # the threshold this detector decided on
            seed=self._config.training.seed,
        )

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
                "best_auroc": r.best_auroc,
                "total_epochs": r.total_epochs,
                "device_used": r.device_used,
                "run_dir": str(r.model_path.parent),
            }
        if self._test_metrics is not None:
            auroc, threshold, image_f1 = self._test_metrics
            result["test"] = {
                "auroc": auroc,
                "threshold": threshold,
                "image_f1": image_f1,
            }
            if self._metric_cis:
                result["test"]["confidence_intervals"] = {
                    name: ci.to_dict() for name, ci in self._metric_cis.items()
                }
        return result

    # ── private ───────────────────────────────────────────────────────────────

    def _render_plots(self, run_dir: Path) -> list[Path]:
        """Render the AUROC-over-epochs curve (reuses the generic plotter curve)."""
        assert self._train_result is not None
        auroc_path = run_dir / "auroc.png"
        epochs = [r.epoch for r in self._train_result.history]
        values = [r.val_auroc for r in self._train_result.history]
        MetricsPlotter.metric_curve(
            epochs,
            values,
            auroc_path,
            label="AUROC",
            ylabel="Image AUROC",
            title="Image-level AUROC over epochs",
        )
        return [auroc_path]

    def _update_run_json(self, run_dir: Path, graphics: list[Path]) -> None:
        """Rewrite run.json with test metrics and artifact paths."""
        run_json_path = run_dir / "run.json"
        if not run_json_path.exists():
            return

        data: dict[str, Any] = json.loads(run_json_path.read_text(encoding="utf-8"))
        if self._test_metrics is not None:
            auroc, threshold, image_f1 = self._test_metrics
            data["metrics"].update(
                {
                    "test_auroc": auroc,
                    "test_threshold": threshold,
                    "test_image_f1": image_f1,
                }
            )
        if self._metric_cis:
            data["metric_cis"] = {
                name: ci.to_dict() for name, ci in self._metric_cis.items()
            }
        data["artifacts"]["graphics"] = [str(p) for p in graphics]
        run_json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


__all__ = ["AnomalyBlock"]
