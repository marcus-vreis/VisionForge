from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from loguru import logger

from visionforge.blocks.base import ExperimentBlock
from visionforge.core.data import DataModule
from visionforge.core.evaluator import EvalResult, Evaluator
from visionforge.core.trainer import Trainer, TrainResult
from visionforge.models.factory import ModelFactory
from visionforge.utils.config import ExperimentConfig


class TransferLearningBlock(ExperimentBlock):
    """Feature extraction and fine-tuning experiment block.

    Supports two modes via config.transfer_learning.mode:
    - "feature_extraction": freeze all backbone layers, train only the head.
    - "fine_tuning": optionally freeze layers before unfreeze_from_layer, then
      use a lower LR for backbone params and full LR for the head.
    """

    def setup(self, config: ExperimentConfig) -> None:
        """Validate config and initialise internal state.

        Raises:
            ValueError: if transfer_learning config is absent.
        """
        if config.transfer_learning is None:
            raise ValueError(
                "TransferLearningBlock requires transfer_learning to be set in ExperimentConfig."
            )
        self._config = config
        self._train_result: TrainResult | None = None
        self._eval_result: EvalResult | None = None
        self._frozen_layers: list[str] = []
        self._optimizer: torch.optim.Optimizer | None = None

    def run(self) -> None:
        """Build model, freeze/unfreeze layers, train, and evaluate."""
        model = ModelFactory.create(self._config.model)
        self._apply_freeze(model)
        self._optimizer = self._build_transfer_optimizer(model)

        data = DataModule(self._config)
        self._train_result = Trainer(self._config).fit(model, data, self._optimizer)

        state_dict = torch.load(
            str(self._train_result.model_path),
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state_dict)  # type: ignore[arg-type]

        self._eval_result = Evaluator(self._config).evaluate(model, data.test_loader())

        run_dir = self._train_result.model_path.parent
        self._update_run_json(run_dir)

    def report(self) -> dict[str, Any]:
        """Return a summary of the run including transfer-learning metadata."""
        if self._train_result is None and self._eval_result is None:
            return {}

        tl = self._config.transfer_learning
        assert tl is not None

        lr_groups: list[dict[str, Any]] = []
        if self._optimizer is not None:
            for g in self._optimizer.param_groups:
                lr_groups.append({"lr": g["lr"], "n_params": len(g["params"])})

        result: dict[str, Any] = {
            "mode": tl.mode,
            "frozen_layers": self._frozen_layers,
            "lr_groups": lr_groups,
        }

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

    # ── freeze / unfreeze ─────────────────────────────────────────────────────

    def _apply_freeze(self, model: nn.Module) -> None:
        """Freeze backbone parameters according to the configured mode.

        For feature_extraction: freeze all layers except the classifier head
        (the last named child of the model).

        For fine_tuning: if unfreeze_from_layer is set, freeze every layer
        that appears before it in named_children(); everything from that layer
        onward is left trainable. If unfreeze_from_layer is None, all layers
        remain trainable.

        Raises:
            ValueError: if unfreeze_from_layer names a layer that does not exist.
        """
        tl = self._config.transfer_learning
        assert tl is not None

        children = list(model.named_children())

        if tl.mode == "feature_extraction":
            # Freeze everything except the final child (classifier head).
            head_name = children[-1][0] if children else None
            for name, child in children:
                if name != head_name:
                    for param in child.parameters():
                        param.requires_grad = False
                    self._frozen_layers.append(name)
                    logger.debug("Froze layer: {}", name)
                else:
                    for param in child.parameters():
                        param.requires_grad = True

        elif tl.mode == "fine_tuning":
            if tl.unfreeze_from_layer is None:
                # All layers trainable — nothing to freeze.
                for param in model.parameters():
                    param.requires_grad = True
                return

            layer_names = [name for name, _ in children]
            if tl.unfreeze_from_layer not in layer_names:
                raise ValueError(
                    f"unfreeze_from_layer='{tl.unfreeze_from_layer}' not found in model. "
                    f"Available layers: {layer_names}"
                )

            # Freeze layers before unfreeze_from_layer, thaw the rest.
            freeze = True
            for name, child in children:
                if name == tl.unfreeze_from_layer:
                    freeze = False
                for param in child.parameters():
                    param.requires_grad = not freeze
                if freeze:
                    self._frozen_layers.append(name)
                    logger.debug("Froze layer: {}", name)

    # ── optimizer construction ────────────────────────────────────────────────

    def _build_transfer_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Build an optimizer with param groups suited to the transfer mode.

        For feature_extraction: single group containing only trainable (head) params.

        For fine_tuning: two groups — unfrozen backbone params at reduced LR,
        head params at the full learning_rate.

        Returns:
            Configured optimizer instance.
        """
        tl = self._config.transfer_learning
        assert tl is not None

        cfg = self._config.training
        builder: Callable[..., torch.optim.Optimizer] = {
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD,
            "adamw": torch.optim.AdamW,
        }[cfg.optimizer]

        if tl.mode == "feature_extraction":
            trainable = [p for p in model.parameters() if p.requires_grad]
            return builder(
                trainable,
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
            )

        # fine_tuning: split into backbone (non-head) and head param groups.
        children = list(model.named_children())
        head_name = children[-1][0] if children else None

        backbone_params: list[nn.Parameter] = []
        head_params: list[nn.Parameter] = []

        for name, child in children:
            params = [p for p in child.parameters() if p.requires_grad]
            if name == head_name:
                head_params.extend(params)
            else:
                backbone_params.extend(params)

        backbone_lr = cfg.learning_rate * tl.backbone_lr_multiplier

        param_groups: list[dict[str, Any]] = [
            {
                "params": backbone_params,
                "lr": backbone_lr,
                "weight_decay": cfg.weight_decay,
            },
            {
                "params": head_params,
                "lr": cfg.learning_rate,
                "weight_decay": cfg.weight_decay,
            },
        ]
        return builder(param_groups)

    # ── run.json ──────────────────────────────────────────────────────────────

    def _update_run_json(self, run_dir: Path) -> None:
        """Rewrite run.json to include transfer-learning metadata and eval metrics."""
        run_json_path = run_dir / "run.json"
        if not run_json_path.exists():
            return

        tl = self._config.transfer_learning
        assert tl is not None

        data: dict[str, Any] = json.loads(run_json_path.read_text(encoding="utf-8"))

        data["transfer_learning"] = {
            "mode": tl.mode,
            "frozen_layers": self._frozen_layers,
            "unfreeze_from_layer": tl.unfreeze_from_layer,
            "backbone_lr_multiplier": tl.backbone_lr_multiplier,
        }

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

        run_json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


__all__ = ["TransferLearningBlock"]
