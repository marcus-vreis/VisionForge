from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from visionforge.blocks.base import ExperimentBlock
from visionforge.core.data import DataModule
from visionforge.core.evaluator import EvalResult, Evaluator, bootstrap_eval_cis
from visionforge.core.metric_ci import MetricCI
from visionforge.core.plotter import MetricsPlotter
from visionforge.core.trainer import Trainer, TrainResult
from visionforge.models.factory import ModelFactory
from visionforge.utils.config import ExperimentConfig


class ClassificationBlock(ExperimentBlock):
    """End-to-end classification experiment block.

    Modes (config.classification.mode):
    - "train": ModelFactory → DataModule → Trainer → Evaluator → all plots → run.json
    - "evaluate": loads checkpoint → Evaluator on test set
    - "infer": single-image inference
    """

    def setup(self, config: ExperimentConfig) -> None:
        self._config = config
        self._train_result: TrainResult | None = None
        self._eval_result: EvalResult | None = None
        self._metric_cis: dict[str, MetricCI] = {}
        # Injected by the GUI layer to stream live epoch progress via SSE.
        self._progress_callback: Callable[[dict[str, Any]], None] | None = None

    def run(self) -> None:
        mode = self._config.classification.mode
        if mode == "train":
            self._run_train()
        elif mode == "evaluate":
            self._run_evaluate()
        elif mode == "infer":
            raise NotImplementedError("Infer mode is not implemented.")

    def report(self) -> dict[str, Any]:
        """Return a summary of the run for logging and GUI display."""
        result: dict[str, Any] = {}
        if self._train_result is not None:
            result["train"] = {
                "best_epoch": self._train_result.best_epoch,
                "best_val_loss": self._train_result.best_val_loss,
                "total_epochs": self._train_result.total_epochs,
                "device_used": self._train_result.device_used,
            }
        if self._eval_result is not None:
            result["eval"] = {
                "accuracy": self._eval_result.accuracy,
                "f1": self._eval_result.f1,
                "precision": self._eval_result.precision,
                "recall": self._eval_result.recall,
                "auc_roc": self._eval_result.auc_roc,
            }
            if self._metric_cis:
                result["eval"]["confidence_intervals"] = {
                    name: ci.to_dict() for name, ci in self._metric_cis.items()
                }
        return result

    # ── private ───────────────────────────────────────────────────────────────

    def _run_train(self) -> None:
        model = ModelFactory.create(self._config.model)
        data = DataModule(self._config)

        # The worker pools have to be stopped even when the run raises: inside
        # `visionforge gui` the traceback keeps `data` alive, so without this the
        # processes outlive the run and each retry stacks more of them.
        try:
            self._train_result = Trainer(self._config).fit(
                model,
                data,
                progress_callback=self._progress_callback,
                cancel_token=self._cancel_token,
                resume_dir=self._resume_dir,
            )

            # Reload best checkpoint for test-set evaluation.
            state_dict = torch.load(
                str(self._train_result.model_path),
                map_location="cpu",
                weights_only=True,
            )
            model.load_state_dict(state_dict)  # type: ignore[arg-type]

            self._evaluate(model, data)

            run_dir = self._train_result.model_path.parent
            graphics = self._render_plots(run_dir, data.class_names)
            self._update_run_json(run_dir, graphics)
        finally:
            data.close()

    def _run_evaluate(self) -> None:
        checkpoint_path = self._config.classification.checkpoint_path
        if checkpoint_path is None:
            raise ValueError("checkpoint_path must be set for mode='evaluate'.")

        model = ModelFactory.create(self._config.model)
        state_dict = torch.load(
            str(checkpoint_path), map_location="cpu", weights_only=True
        )
        model.load_state_dict(state_dict)  # type: ignore[arg-type]

        data = DataModule(self._config)
        try:
            self._evaluate(model, data)
        finally:
            data.close()

    def _evaluate(self, model: Any, data: DataModule) -> None:
        """Score the test split and attach a bootstrap CI to each metric."""
        result = Evaluator(self._config).evaluate(model, data.test_loader())
        self._eval_result = result
        self._metric_cis = bootstrap_eval_cis(result, self._config)

    def _render_plots(self, run_dir: Path, class_names: list[str]) -> list[Path]:
        """Render every available plot for the current train + eval results.

        Returns the ordered list of plot paths that were actually written. Plots
        that are undefined (e.g. ROC with a single class) are silently skipped.
        """
        assert self._train_result is not None
        graphics: list[Path] = []

        loss_path = run_dir / "loss.png"
        MetricsPlotter.loss_curve(self._train_result.history, loss_path)
        graphics.append(loss_path)

        acc_path = run_dir / "accuracy.png"
        MetricsPlotter.accuracy_curve(self._train_result.history, acc_path)
        graphics.append(acc_path)

        if self._eval_result is None:
            return graphics

        cm_path = run_dir / "confusion_matrix.png"
        MetricsPlotter.confusion_matrix_plot(
            self._eval_result.confusion_matrix, class_names, cm_path
        )
        graphics.append(cm_path)

        cm_norm_path = run_dir / "confusion_matrix_normalized.png"
        MetricsPlotter.confusion_matrix_normalized(
            self._eval_result.confusion_matrix, class_names, cm_norm_path
        )
        graphics.append(cm_norm_path)

        roc_path = run_dir / "roc_curve.png"
        if MetricsPlotter.roc_curve_plot(
            self._eval_result.y_true,
            self._eval_result.y_proba_full,
            class_names,
            roc_path,
        ):
            graphics.append(roc_path)

        pr_path = run_dir / "precision_recall_curve.png"
        if MetricsPlotter.precision_recall_curve_plot(
            self._eval_result.y_true,
            self._eval_result.y_proba_full,
            class_names,
            pr_path,
        ):
            graphics.append(pr_path)

        return graphics

    def _update_run_json(self, run_dir: Path, graphics: list[Path]) -> None:
        """Rewrite run.json with full metrics and artifact paths."""
        run_json_path = run_dir / "run.json"
        if not run_json_path.exists():
            return

        data: dict[str, Any] = json.loads(run_json_path.read_text(encoding="utf-8"))

        if self._eval_result is not None:
            data["metrics"].update(
                {
                    "test_accuracy": self._eval_result.accuracy,
                    "test_f1": self._eval_result.f1,
                    "test_precision": self._eval_result.precision,
                    "test_recall": self._eval_result.recall,
                    "test_auc_roc": self._eval_result.auc_roc,
                }
            )
        if self._metric_cis:
            # Sibling of "metrics" rather than a nested entry inside it: several
            # readers treat metrics as a flat name -> number map (the history
            # projection, the markdown table), and a dict value there would
            # either be skipped or printed raw.
            data["metric_cis"] = {
                name: ci.to_dict() for name, ci in self._metric_cis.items()
            }
        data["artifacts"]["graphics"] = [str(p) for p in graphics]

        run_json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


__all__ = ["ClassificationBlock"]
