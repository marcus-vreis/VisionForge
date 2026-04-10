from __future__ import annotations

from dataclasses import dataclass
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

from visionforge.utils.config import ExperimentConfig


@dataclass
class EvalResult:
    """Metrics computed on a dataset split."""

    accuracy: float
    f1: float
    precision: float
    recall: float
    auc_roc: float | None  # None for multiclass (requires per-class probabilities)
    confusion_matrix: list[list[int]]
    report: str  # sklearn classification_report text


class Evaluator:
    """Computes classification metrics on a data loader."""

    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate(self, model: nn.Module, loader: Any) -> EvalResult:
        """Run inference on loader and return classification metrics.

        Args:
            model: trained nn.Module.
            loader: iterable of (inputs, labels) batches.

        Returns:
            EvalResult with accuracy, F1, precision, recall, AUC-ROC, confusion matrix.
        """
        model.eval()
        model = model.to(self._device)

        all_preds: list[int] = []
        all_labels: list[int] = []
        all_probs: list[float] = []

        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self._device)
                outputs = model(inputs)

                if self._config.task == "binary":
                    probs = outputs.sigmoid().squeeze(1)
                    preds = (probs > 0.5).long()
                else:
                    probs = outputs.softmax(dim=1).max(dim=1).values
                    preds = outputs.argmax(dim=1)

                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.tolist())
                all_probs.extend(probs.cpu().tolist())

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)

        avg = "binary" if self._config.task == "binary" else "macro"
        auc_roc: float | None = None

        if self._config.task == "binary" and len(np.unique(y_true)) == 2:
            auc_roc = float(roc_auc_score(y_true, y_prob))

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
        )


__all__ = ["Evaluator", "EvalResult"]
