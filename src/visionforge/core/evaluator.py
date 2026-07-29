from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from visionforge.core.metric_ci import MetricCI, bootstrap_classification_cis
from visionforge.core.trainer import resolve_device
from visionforge.utils.config import ExperimentConfig


@dataclass
class EvalResult:
    """Metrics computed on a dataset split."""

    accuracy: float
    f1: float
    precision: float
    recall: float
    auc_roc: float | None  # None for multiclass without per-class probabilities
    confusion_matrix: list[list[int]]
    report: str  # sklearn classification_report text
    # Per-sample arrays kept for downstream plotting (ROC, PR, calibration).
    # For binary tasks: y_score is sigmoid prob of positive class.
    # For multiclass: y_score is softmax prob of the predicted class.
    y_true: list[int] = field(default_factory=list)
    y_score: list[float] = field(default_factory=list)
    # Full per-class softmax matrix for multiclass ROC (one row per sample).
    y_proba_full: list[list[float]] = field(default_factory=list)
    # Predicted labels. Kept alongside the probabilities because a multiclass
    # y_score only holds the winning class's probability, which cannot recover
    # which class won — and the bootstrap CIs (ADR-074) recompute metrics from
    # the predictions, not from the aggregates.
    y_pred: list[int] = field(default_factory=list)


class Evaluator:
    """Computes classification metrics on a data loader."""

    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config
        self._device, self._dp_ids, _ = resolve_device(config.device)

    def evaluate(self, model: nn.Module, loader: Any) -> EvalResult:
        """Run inference on loader and return classification metrics.

        Args:
            model: trained nn.Module.
            loader: iterable of (inputs, labels) batches.

        Returns:
            EvalResult with metrics, confusion matrix, and per-sample arrays for plotting.
        """
        model.eval()
        model = model.to(self._device)

        all_preds: list[int] = []
        all_labels: list[int] = []
        all_probs: list[float] = []  # prob of predicted class (binary: positive class)
        all_proba_full: list[list[float]] = []

        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self._device, non_blocking=True)
                outputs = model(inputs)

                if self._config.task == "binary":
                    pos = outputs.sigmoid().squeeze(1)
                    preds = (pos > 0.5).long()
                    all_probs.extend(pos.cpu().tolist())
                    # For binary, full proba is [1-p, p] per sample.
                    all_proba_full.extend(
                        [[1.0 - float(p), float(p)] for p in pos.cpu().tolist()]
                    )
                else:
                    softmax = outputs.softmax(dim=1)
                    preds = softmax.argmax(dim=1)
                    all_probs.extend(softmax.max(dim=1).values.cpu().tolist())
                    all_proba_full.extend(softmax.cpu().tolist())

                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.tolist())

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)

        avg = "binary" if self._config.task == "binary" else "macro"
        auc_roc: float | None = None

        if self._config.task == "binary" and len(np.unique(y_true)) == 2:
            auc_roc = float(roc_auc_score(y_true, y_prob))
        elif self._config.task == "multiclass" and len(all_proba_full) > 0:
            n_classes = len(all_proba_full[0])
            present = np.unique(y_true)
            if len(present) > 1 and n_classes >= 2:
                try:
                    auc_roc = float(
                        roc_auc_score(
                            y_true,
                            np.array(all_proba_full),
                            multi_class="ovr",
                            average="macro",
                        )
                    )
                except ValueError:
                    auc_roc = None

        return EvalResult(
            accuracy=float(accuracy_score(y_true, y_pred)),
            f1=float(f1_score(y_true, y_pred, average=avg, zero_division=0)),
            precision=float(
                precision_score(y_true, y_pred, average=avg, zero_division=0)
            ),
            recall=float(recall_score(y_true, y_pred, average=avg, zero_division=0)),
            auc_roc=auc_roc,
            confusion_matrix=confusion_matrix(y_true, y_pred).tolist(),
            report=classification_report(y_true, y_pred, zero_division=0),  # type: ignore[call-arg]
            y_true=all_labels,
            y_score=all_probs,
            y_proba_full=all_proba_full,
            y_pred=all_preds,
        )


def bootstrap_eval_cis(
    result: EvalResult, config: ExperimentConfig
) -> dict[str, MetricCI]:
    """Bootstrap CIs for an evaluated split, seeded from the run's own seed (ADR-074).

    Lives here rather than in each block so every classification path that
    evaluates a test set reports the interval the same way — and derives it from
    ``training.seed``, so re-reading a finished run never yields a different
    interval than the one already written to ``run.json``.
    """
    return bootstrap_classification_cis(
        result.y_true,
        result.y_pred,
        task=config.task,
        y_proba_full=result.y_proba_full,
        seed=config.training.seed,
    )


__all__ = ["Evaluator", "EvalResult", "bootstrap_eval_cis"]
