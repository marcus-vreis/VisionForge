"""Titles, axes and legends of the saved figures follow the run's language (ADR-116).

The figure is read off the object that gets saved, not off the image: what is
being checked is the text that was set, and a pixel comparison would say nothing
about which language it is in.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from matplotlib.figure import Figure

from visionforge.core.plotter import MetricsPlotter
from visionforge.utils.messages import using_lang

from .test_plotter import make_history

# Title, x label, y label and legend entries of one axes.
Panel = tuple[str, str, str, list[str]]


@pytest.fixture
def figures(monkeypatch: pytest.MonkeyPatch) -> list[Figure]:
    """Every figure a plotter hands to ``savefig``, without rendering it."""
    drawn: list[Figure] = []
    monkeypatch.setattr(
        Figure, "savefig", lambda self, *args, **kwargs: drawn.append(self)
    )
    return drawn


def _panels(fig: Figure) -> list[Panel]:
    out: list[Panel] = []
    for ax in fig.axes:
        # A heatmap's colour bar is an axes of its own, with no title.
        if not ax.get_title():
            continue
        legend = ax.get_legend()
        entries = [t.get_text() for t in legend.get_texts()] if legend else []
        out.append((ax.get_title(), ax.get_xlabel(), ax.get_ylabel(), entries))
    return out


_PROBA = [[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]]
_PROBA3 = [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.7, 0.2, 0.1]]

# (name, how to draw it, what it says in Portuguese, what it says in English)
_CASES: list[tuple[str, Callable[[Path], Any], list[Panel], list[Panel]]] = [
    (
        "loss_curve",
        lambda p: MetricsPlotter.loss_curve(make_history(), p),
        [("Perda de treino e de validação", "Época", "Perda", ["treino", "validação"])],
        [("Training & Validation Loss", "Epoch", "Loss", ["train", "val"])],
    ),
    (
        "accuracy_curve",
        lambda p: MetricsPlotter.accuracy_curve(make_history(), p),
        [
            (
                "Acurácia de treino e de validação",
                "Época",
                "Acurácia",
                ["treino", "validação"],
            )
        ],
        [("Training & Validation Accuracy", "Epoch", "Accuracy", ["train", "val"])],
    ),
    (
        "metric_curve",
        lambda p: MetricsPlotter.metric_curve([1, 2], [0.5, 0.6], p, label="AUROC"),
        [("AUROC por época", "Época", "AUROC", ["AUROC"])],
        [("AUROC over epochs", "Epoch", "AUROC", ["AUROC"])],
    ),
    (
        "detection_results",
        lambda p: MetricsPlotter.detection_results(
            [1, 2], [1.0, 0.8], [1.1, 0.9], [0.2, 0.4], p
        ),
        [
            (
                "Perda de treino e de validação",
                "Época",
                "Perda",
                ["treino", "validação"],
            ),
            ("mAP@50 de validação", "Época", "mAP@50", ["mAP@50"]),
        ],
        [
            ("Training & Validation Loss", "Epoch", "Loss", ["train", "val"]),
            ("Validation mAP@50", "Epoch", "mAP@50", ["mAP@50"]),
        ],
    ),
    (
        "regression_scatter",
        lambda p: MetricsPlotter.regression_scatter(
            [1.0, 2.0], [1.1, 1.9], p, target_name="price"
        ),
        [("Predito vs real", "real (price)", "predito (price)", ["perfeito (y = ŷ)"])],
        [
            (
                "Predicted vs actual",
                "actual (price)",
                "predicted (price)",
                ["perfect (y = ŷ)"],
            )
        ],
    ),
    (
        "residual_histogram",
        lambda p: MetricsPlotter.residual_histogram([1.0, 2.0], [1.5, 2.5], p),
        [
            (
                "Distribuição dos resíduos",
                "resíduo (predito − real)",
                "amostras",
                ["sem erro", "viés médio = +0.5000"],
            )
        ],
        [
            (
                "Residual distribution",
                "residual (predicted − actual)",
                "samples",
                ["no error", "mean bias = +0.5000"],
            )
        ],
    ),
    (
        "per_class_bars",
        lambda p: MetricsPlotter.per_class_bars([0.4, 0.7], ["a", "b"], p),
        [("IoU por classe", "", "IoU", ["média = 0.5500"])],
        [("IoU per class", "", "IoU", ["mean = 0.5500"])],
    ),
    (
        "score_histogram",
        lambda p: MetricsPlotter.score_histogram(
            [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], p, threshold=0.5
        ),
        [
            (
                "Distribuição dos escores",
                "escore de anomalia",
                "imagens",
                ["normal (n=2)", "defeito (n=2)", "limiar = 0.5000"],
            )
        ],
        [
            (
                "Score distribution",
                "anomaly score",
                "images",
                ["normal (n=2)", "defect (n=2)", "threshold = 0.5000"],
            )
        ],
    ),
    (
        "confusion_matrix_plot",
        lambda p: MetricsPlotter.confusion_matrix_plot([[3, 1], [0, 4]], ["a", "b"], p),
        [("Matriz de confusão", "Predito", "Real", [])],
        [("Confusion Matrix", "Predicted", "True", [])],
    ),
    (
        "confusion_matrix_normalized",
        lambda p: MetricsPlotter.confusion_matrix_normalized(
            [[3, 1], [0, 4]], ["a", "b"], p
        ),
        [
            (
                "Matriz de confusão (normalizada pela classe real)",
                "Predito",
                "Real",
                [],
            )
        ],
        [("Confusion Matrix (normalized by true class)", "Predicted", "True", [])],
    ),
    (
        "roc_curve_plot",
        lambda p: MetricsPlotter.roc_curve_plot([0, 1, 0, 1], _PROBA, ["a", "b"], p),
        [
            (
                "Curva ROC",
                "Taxa de falsos positivos",
                "Taxa de verdadeiros positivos",
                ["ROC (AUC = 1.000)", "acaso"],
            )
        ],
        [
            (
                "ROC Curve",
                "False Positive Rate",
                "True Positive Rate",
                ["ROC (AUC = 1.000)", "chance"],
            )
        ],
    ),
    (
        "roc_curve_plot_multiclass",
        # Two names for three classes: the third is called by its index.
        lambda p: MetricsPlotter.roc_curve_plot([0, 1, 2, 0], _PROBA3, ["a", "b"], p),
        [
            (
                "Curva ROC",
                "Taxa de falsos positivos",
                "Taxa de verdadeiros positivos",
                [
                    "a (AUC = 1.000)",
                    "b (AUC = 1.000)",
                    "classe 2 (AUC = 1.000)",
                    "macro (AUC = 1.000)",
                    "acaso",
                ],
            )
        ],
        [
            (
                "ROC Curve",
                "False Positive Rate",
                "True Positive Rate",
                [
                    "a (AUC = 1.000)",
                    "b (AUC = 1.000)",
                    "class 2 (AUC = 1.000)",
                    "macro (AUC = 1.000)",
                    "chance",
                ],
            )
        ],
    ),
    (
        "precision_recall_curve_plot",
        lambda p: MetricsPlotter.precision_recall_curve_plot(
            [0, 1, 0, 1], _PROBA, ["a", "b"], p
        ),
        [
            (
                "Curva de precisão e revocação",
                "Revocação",
                "Precisão",
                ["PR (AUC = 1.000)"],
            )
        ],
        [("Precision-Recall Curve", "Recall", "Precision", ["PR (AUC = 1.000)"])],
    ),
]


@pytest.mark.parametrize(
    ("draw", "portuguese", "english"),
    [pytest.param(c[1], c[2], c[3], id=c[0]) for c in _CASES],
)
class TestFiguresFollowTheLanguage:
    def _drawn(
        self, draw: Callable[[Path], Any], tmp_path: Path, figures: list[Figure]
    ) -> list[Panel]:
        draw(tmp_path / "plot.png")
        assert len(figures) == 1
        return _panels(figures[0])

    def test_portuguese_without_a_bound_language(
        self,
        draw: Callable[[Path], Any],
        portuguese: list[Panel],
        english: list[Panel],
        tmp_path: Path,
        figures: list[Figure],
    ) -> None:
        assert self._drawn(draw, tmp_path, figures) == portuguese

    def test_english_under_the_english_interface(
        self,
        draw: Callable[[Path], Any],
        portuguese: list[Panel],
        english: list[Panel],
        tmp_path: Path,
        figures: list[Figure],
    ) -> None:
        with using_lang("en"):
            assert self._drawn(draw, tmp_path, figures) == english


def test_a_title_the_caller_passes_is_kept_as_given(
    tmp_path: Path, figures: list[Figure]
) -> None:
    with using_lang("en"):
        MetricsPlotter.metric_curve([1], [0.5], tmp_path / "a.png", title="Mine")
        MetricsPlotter.per_class_bars([0.5], ["a"], tmp_path / "b.png", title="Mine")

    assert [fig.axes[0].get_title() for fig in figures] == ["Mine", "Mine"]
