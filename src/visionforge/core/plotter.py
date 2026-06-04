from __future__ import annotations

from pathlib import Path

import numpy as np
import seaborn as sns
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from visionforge.core.trainer import EpochResult


class MetricsPlotter:
    """Generates and saves metric plots for a training run.

    Each method is independent — add new plot types without touching existing ones.
    Figures are rendered via the Agg backend so no display is required.
    """

    @staticmethod
    def loss_curve(history: list[EpochResult], save_path: Path) -> None:
        """Save a train/val loss curve to save_path."""
        epochs = [r.epoch for r in history]
        train_losses = [r.train_loss for r in history]
        val_losses = [r.val_loss for r in history]

        fig = Figure(figsize=(10, 6))
        FigureCanvasAgg(fig)
        ax = fig.subplots()

        ax.plot(epochs, train_losses, label="train", linewidth=2, color="#3b82f6")
        ax.plot(epochs, val_losses, label="val", linewidth=2, color="#f59e0b")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training & Validation Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    @staticmethod
    def metric_curve(
        epochs: list[int],
        values: list[float],
        save_path: Path,
        *,
        label: str = "metric",
        ylabel: str | None = None,
        title: str | None = None,
        color: str = "#8b5cf6",
    ) -> None:
        """Save a single-series metric-over-epochs curve (e.g. AUROC).

        Generic so any task can plot one streamed metric without a bespoke
        method; used by the anomaly block for the AUROC curve.
        """
        fig = Figure(figsize=(10, 6))
        FigureCanvasAgg(fig)
        ax = fig.subplots()

        ax.plot(epochs, values, label=label, linewidth=2, color=color, marker="o")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel or label)
        ax.set_title(title or f"{label} over epochs")
        ax.legend()
        ax.grid(True, alpha=0.3)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    @staticmethod
    def accuracy_curve(history: list[EpochResult], save_path: Path) -> None:
        """Save a train/val accuracy curve to save_path."""
        epochs = [r.epoch for r in history]
        train_accs = [r.train_accuracy for r in history]
        val_accs = [r.val_accuracy for r in history]

        fig = Figure(figsize=(10, 6))
        FigureCanvasAgg(fig)
        ax = fig.subplots()

        ax.plot(epochs, train_accs, label="train", linewidth=2, color="#10b981")
        ax.plot(epochs, val_accs, label="val", linewidth=2, color="#8b5cf6")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Training & Validation Accuracy")
        ax.set_ylim(0.0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    @staticmethod
    def confusion_matrix_plot(
        cm: list[list[int]],
        class_names: list[str],
        save_path: Path,
    ) -> None:
        """Save a seaborn confusion matrix heatmap to save_path."""
        cm_arr = np.array(cm)

        fig = Figure(figsize=(8, 6))
        FigureCanvasAgg(fig)
        ax = fig.subplots()

        sns.heatmap(
            cm_arr,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")

        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    @staticmethod
    def confusion_matrix_normalized(
        cm: list[list[int]],
        class_names: list[str],
        save_path: Path,
    ) -> None:
        """Save a row-normalized confusion matrix (per-class recall) heatmap."""
        cm_arr = np.array(cm, dtype=float)
        # Row-normalize; rows that sum to zero stay zero.
        row_sums = cm_arr.sum(axis=1, keepdims=True)
        normalized = np.divide(
            cm_arr, row_sums, out=np.zeros_like(cm_arr), where=row_sums != 0
        )

        fig = Figure(figsize=(8, 6))
        FigureCanvasAgg(fig)
        ax = fig.subplots()

        sns.heatmap(
            normalized,
            annot=True,
            fmt=".2f",
            cmap="Purples",
            xticklabels=class_names,
            yticklabels=class_names,
            vmin=0.0,
            vmax=1.0,
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix (normalized by true class)")

        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    @staticmethod
    def roc_curve_plot(
        y_true: list[int],
        y_proba_full: list[list[float]],
        class_names: list[str],
        save_path: Path,
    ) -> bool:
        """Save a ROC curve plot. Returns False (without saving) when undefined.

        Binary: single curve using positive-class probability.
        Multiclass: one curve per class via one-vs-rest, plus macro-average.
        Returns False if only one class is present (ROC undefined).
        """
        y = np.array(y_true)
        if len(np.unique(y)) < 2:
            return False
        proba = np.array(y_proba_full)
        if proba.ndim != 2 or proba.shape[0] != len(y):
            return False

        fig = Figure(figsize=(8, 6))
        FigureCanvasAgg(fig)
        ax = fig.subplots()

        n_classes = proba.shape[1]
        if n_classes == 2:
            fpr, tpr, _ = roc_curve(y, proba[:, 1])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, linewidth=2, label=f"ROC (AUC = {roc_auc:.3f})")
        else:
            # One-vs-rest per class.
            for i in range(n_classes):
                y_bin = (y == i).astype(int)
                if y_bin.sum() == 0:
                    continue
                fpr, tpr, _ = roc_curve(y_bin, proba[:, i])
                roc_auc = auc(fpr, tpr)
                label = class_names[i] if i < len(class_names) else f"class {i}"
                ax.plot(fpr, tpr, linewidth=1.6, label=f"{label} (AUC = {roc_auc:.3f})")

        ax.plot([0, 1], [0, 1], "--", color="#94a3b8", linewidth=1, label="chance")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.02)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return True

    @staticmethod
    def precision_recall_curve_plot(
        y_true: list[int],
        y_proba_full: list[list[float]],
        class_names: list[str],
        save_path: Path,
    ) -> bool:
        """Save a precision-recall curve. Returns False if undefined."""
        y = np.array(y_true)
        if len(np.unique(y)) < 2:
            return False
        proba = np.array(y_proba_full)
        if proba.ndim != 2 or proba.shape[0] != len(y):
            return False

        fig = Figure(figsize=(8, 6))
        FigureCanvasAgg(fig)
        ax = fig.subplots()

        n_classes = proba.shape[1]
        if n_classes == 2:
            precision, recall, _ = precision_recall_curve(y, proba[:, 1])
            pr_auc = auc(recall, precision)
            ax.plot(recall, precision, linewidth=2, label=f"PR (AUC = {pr_auc:.3f})")
        else:
            for i in range(n_classes):
                y_bin = (y == i).astype(int)
                if y_bin.sum() == 0:
                    continue
                precision, recall, _ = precision_recall_curve(y_bin, proba[:, i])
                pr_auc = auc(recall, precision)
                label = class_names[i] if i < len(class_names) else f"class {i}"
                ax.plot(
                    recall,
                    precision,
                    linewidth=1.6,
                    label=f"{label} (AUC = {pr_auc:.3f})",
                )

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.02)
        ax.legend(loc="lower left", fontsize=9)
        ax.grid(True, alpha=0.3)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return True


__all__ = ["MetricsPlotter"]
