from __future__ import annotations

from pathlib import Path

import numpy as np
import seaborn as sns
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from visionforge.core.trainer import EpochResult


class MetricsPlotter:
    """Generates and saves metric plots for a training run.

    Each method is independent — add new plot types without touching existing ones.
    Figures are rendered via the Agg backend so no display is required.
    """

    @staticmethod
    def loss_curve(history: list[EpochResult], save_path: Path) -> None:
        """Save a train/val loss curve to save_path.

        Args:
            history: list of per-epoch results from Trainer.fit().
            save_path: destination .png path (parent dir created if needed).
        """
        epochs = [r.epoch for r in history]
        train_losses = [r.train_loss for r in history]
        val_losses = [r.val_loss for r in history]

        fig = Figure(figsize=(10, 6))
        FigureCanvasAgg(fig)
        ax = fig.subplots()

        ax.plot(epochs, train_losses, label="train", linewidth=2)
        ax.plot(epochs, val_losses, label="val", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
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
        """Save a seaborn confusion matrix heatmap to save_path.

        Args:
            cm: square confusion matrix as list of lists.
            class_names: ordered class labels for axis ticks.
            save_path: destination .png path (parent dir created if needed).
        """
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


__all__ = ["MetricsPlotter"]
