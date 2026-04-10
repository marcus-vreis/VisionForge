from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from visionforge.blocks.base import ExperimentBlock
from visionforge.core.data import DataModule
from visionforge.core.evaluator import EvalResult, Evaluator
from visionforge.core.plotter import MetricsPlotter
from visionforge.core.trainer import Trainer, TrainResult
from visionforge.models.factory import ModelFactory
from visionforge.utils.config import ExperimentConfig


class ClassificationBlock(ExperimentBlock):
    """End-to-end classification experiment block.

    Supports multiple operating modes via config.classification.mode:
    - "train": ModelFactory → DataModule → Trainer → Evaluator → Plotter → run.json
    - "evaluate": loads checkpoint → Evaluator on test set
    - "infer": loads checkpoint → single-image inference (Phase 4)
    """

    def setup(self, config: ExperimentConfig) -> None:
        self._config = config
        self._train_result: TrainResult | None = None
        self._eval_result: EvalResult | None = None

    def run(self) -> None:
        mode = self._config.classification.mode
        if mode == "train":
            self._run_train()
        elif mode == "evaluate":
            self._run_evaluate()
        elif mode == "infer":
            raise NotImplementedError("Infer mode will be implemented in Phase 4.")

    def report(self) -> dict[str, Any]:
        """Return a summary of the run for logging and GUI display."""
        result: dict[str, Any] = {}
        if self._train_result is not None:
            result["train"] = {
                "best_epoch": self._train_result.best_epoch,
                "best_val_loss": self._train_result.best_val_loss,
                "total_epochs": self._train_result.total_epochs,
            }
        if self._eval_result is not None:
            result["eval"] = {
                "accuracy": self._eval_result.accuracy,
                "f1": self._eval_result.f1,
                "precision": self._eval_result.precision,
                "recall": self._eval_result.recall,
                "auc_roc": self._eval_result.auc_roc,
            }
        return result

    # ── private ───────────────────────────────────────────────────────────────

    def _run_train(self) -> None:
        model = ModelFactory.create(self._config.model)
        data = DataModule(self._config)

        self._train_result = Trainer(self._config).fit(model, data)

        # Reload best checkpoint for test-set evaluation.
        state_dict = torch.load(
            str(self._train_result.model_path),
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state_dict)  # type: ignore[arg-type]

        self._eval_result = Evaluator(self._config).evaluate(model, data.test_loader())

        run_dir = self._train_result.model_path.parent
        loss_path = run_dir / "loss.png"
        cm_path = run_dir / "confusion_matrix.png"

        MetricsPlotter.loss_curve(self._train_result.history, loss_path)
        MetricsPlotter.confusion_matrix_plot(
            self._eval_result.confusion_matrix,
            data.class_names,
            cm_path,
        )

        self._update_run_json(run_dir, loss_path, cm_path)

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
        self._eval_result = Evaluator(self._config).evaluate(model, data.test_loader())

    def _update_run_json(
        self,
        run_dir: Path,
        loss_path: Path,
        cm_path: Path,
    ) -> None:
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
        data["artifacts"]["graphics"] = [str(loss_path), str(cm_path)]

        run_json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


__all__ = ["ClassificationBlock"]
