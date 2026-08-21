from __future__ import annotations

from pathlib import Path

import numpy as np
import seaborn as sns
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
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
        # There is no epoch 1.5, and a figure that implies one invites the
        # question in a review.
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
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
    def regression_scatter(
        y_true: list[float] | np.ndarray,
        y_pred: list[float] | np.ndarray,
        save_path: Path,
        *,
        target_name: str = "target",
    ) -> None:
        """Predicted vs actual, against the identity line (ADR-077).

        The regression counterpart of a confusion matrix: a scatter that hugs the
        diagonal is a good model, and the *shape* of the departure says which
        kind of wrong it is — a flat cloud means the model predicts the mean, a
        tilted one means systematic under- or over-estimation. An error number
        alone cannot distinguish those.
        """
        true = np.asarray(y_true, dtype=float).ravel()
        pred = np.asarray(y_pred, dtype=float).ravel()

        fig = Figure(figsize=(7, 7))
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        ax.scatter(true, pred, s=14, alpha=0.45, color="#5b9fff", edgecolors="none")

        if true.size:
            low = float(min(true.min(), pred.min()))
            high = float(max(true.max(), pred.max()))
            ax.plot(
                [low, high],
                [low, high],
                "--",
                color="#f16363",
                linewidth=1.5,
                label="perfeito (y = ŷ)",
            )
            ax.set_xlim(low, high)
            ax.set_ylim(low, high)
            ax.legend()

        ax.set_xlabel(f"real ({target_name})")
        ax.set_ylabel(f"predito ({target_name})")
        ax.set_title("Predito vs real")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    @staticmethod
    def residual_histogram(
        y_true: list[float] | np.ndarray,
        y_pred: list[float] | np.ndarray,
        save_path: Path,
    ) -> None:
        """Distribution of ``pred - true`` (ADR-077).

        Centred and symmetric is healthy; a shifted centre is bias the aggregate
        error hides, because MAE and RMSE are the same whether the model reads
        every sample too high or scatters both ways.
        """
        residual = (
            np.asarray(y_pred, dtype=float).ravel()
            - np.asarray(y_true, dtype=float).ravel()
        )

        fig = Figure(figsize=(9, 5))
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        ax.hist(residual, bins=40, color="#5b9fff", alpha=0.85)
        ax.axvline(
            0.0, color="#f16363", linestyle="--", linewidth=1.5, label="sem erro"
        )
        if residual.size:
            ax.axvline(
                float(residual.mean()),
                color="#f5a524",
                linewidth=1.5,
                label=f"viés médio = {residual.mean():+.4f}",
            )
        ax.set_xlabel("resíduo (predito − real)")
        ax.set_ylabel("amostras")
        ax.set_title("Distribuição dos resíduos")
        ax.legend()
        ax.grid(True, alpha=0.3)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    @staticmethod
    def per_class_bars(
        values: list[float] | np.ndarray,
        class_names: list[str],
        save_path: Path,
        *,
        metric_label: str = "IoU",
        title: str | None = None,
    ) -> None:
        """One bar per class (ADR-077).

        A mean IoU of 0.55 can be five mediocre classes or four good ones and a
        class the model never finds; only the per-class view separates them, and
        it is the first thing anyone asks of a segmentation result.
        """
        scores = np.asarray(values, dtype=float).ravel()
        names = list(class_names) or [str(i) for i in range(scores.size)]

        fig = Figure(figsize=(max(7.0, 0.7 * scores.size + 3), 5))
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        ax.bar(names, scores, color="#b079ff", alpha=0.9)
        if scores.size:
            ax.axhline(
                float(scores.mean()),
                color="#f5a524",
                linestyle="--",
                linewidth=1.5,
                label=f"média = {scores.mean():.4f}",
            )
            ax.legend()
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel(metric_label)
        ax.set_title(title or f"{metric_label} por classe")
        ax.grid(True, axis="y", alpha=0.3)
        for tick in ax.get_xticklabels():
            tick.set_rotation(30)
            tick.set_horizontalalignment("right")

        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    @staticmethod
    def score_histogram(
        labels: list[int] | np.ndarray,
        scores: list[float] | np.ndarray,
        save_path: Path,
        *,
        threshold: float | None = None,
    ) -> None:
        """Anomaly scores for normal vs defective images, with the threshold (ADR-077).

        AUROC says how separable the two populations are; this says *where* they
        sit and what the chosen cut actually keeps. A researcher moving the
        threshold needs to see the overlap, not a single number summarising it.
        """
        label_arr = np.asarray(labels).ravel()
        score_arr = np.asarray(scores, dtype=float).ravel()

        fig = Figure(figsize=(9, 5))
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        for value, name, color in ((0, "normal", "#48cf8e"), (1, "defeito", "#f16363")):
            subset = score_arr[label_arr == value]
            if subset.size:
                ax.hist(
                    subset,
                    bins=40,
                    alpha=0.6,
                    label=f"{name} (n={subset.size})",
                    color=color,
                )
        if threshold is not None:
            ax.axvline(
                float(threshold),
                color="#f5a524",
                linestyle="--",
                linewidth=1.8,
                label=f"limiar = {threshold:.4f}",
            )
        ax.set_xlabel("escore de anomalia")
        ax.set_ylabel("imagens")
        ax.set_title("Distribuição dos escores")
        ax.legend()
        ax.grid(True, alpha=0.3)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    @staticmethod
    def detection_results(
        epochs: list[int],
        train_losses: list[float | None],
        val_losses: list[float | None],
        map50s: list[float | None],
        save_path: Path,
    ) -> None:
        """Save a detection history figure: train/val loss + mAP@50 over epochs.

        The torchvision backend has no Ultralytics ``results.png``, so this gives
        its runs the same at-a-glance history chart the GUI shows for YOLO runs.
        ``None`` points (a metric not yet available for an epoch) are dropped.
        """

        def _clean(
            xs: list[int], ys: list[float | None]
        ) -> tuple[list[int], list[float]]:
            kept = [(x, y) for x, y in zip(xs, ys, strict=False) if y is not None]
            return [x for x, _ in kept], [y for _, y in kept]

        fig = Figure(figsize=(12, 5))
        FigureCanvasAgg(fig)
        ax_loss, ax_map = fig.subplots(1, 2)

        te, tl = _clean(epochs, train_losses)
        ve, vl = _clean(epochs, val_losses)
        if tl:
            ax_loss.plot(te, tl, label="train", linewidth=2, color="#3b82f6")
        if vl:
            ax_loss.plot(ve, vl, label="val", linewidth=2, color="#f59e0b")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_title("Training & Validation Loss")
        if tl or vl:
            ax_loss.legend()
        ax_loss.grid(True, alpha=0.3)

        me, mv = _clean(epochs, map50s)
        if mv:
            ax_map.plot(
                me, mv, label="mAP@50", linewidth=2, color="#10b981", marker="o"
            )
            ax_map.legend()
        ax_map.set_xlabel("Epoch")
        ax_map.set_ylabel("mAP@50")
        ax_map.set_title("Validation mAP@50")
        ax_map.grid(True, alpha=0.3)

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
            # One-vs-rest per class, plus the macro average over them — the
            # single number `test_auc_roc` records, which until now appeared
            # nowhere on the figure that is supposed to show it.
            grid = np.linspace(0.0, 1.0, 201)
            interpolated: list[np.ndarray] = []
            for i in range(n_classes):
                y_bin = (y == i).astype(int)
                if y_bin.sum() == 0:
                    continue
                fpr, tpr, _ = roc_curve(y_bin, proba[:, i])
                roc_auc = auc(fpr, tpr)
                label = class_names[i] if i < len(class_names) else f"class {i}"
                ax.plot(fpr, tpr, linewidth=1.6, label=f"{label} (AUC = {roc_auc:.3f})")
                interpolated.append(np.interp(grid, fpr, tpr))
            if len(interpolated) > 1:
                macro = np.mean(interpolated, axis=0)
                ax.plot(
                    grid,
                    macro,
                    linewidth=2.4,
                    color="#111827",
                    linestyle=":",
                    label=f"macro (AUC = {auc(grid, macro):.3f})",
                )

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
