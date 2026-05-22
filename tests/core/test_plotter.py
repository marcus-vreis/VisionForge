from pathlib import Path

from visionforge.core.plotter import MetricsPlotter
from visionforge.core.trainer import EpochResult


def make_history(n: int = 5) -> list[EpochResult]:
    return [
        EpochResult(
            epoch=i,
            train_loss=1.0 / i,
            train_accuracy=0.4 + i * 0.05,
            val_loss=1.2 / i,
            val_accuracy=0.5 + i * 0.05,
        )
        for i in range(1, n + 1)
    ]


class TestLossCurve:
    def test_creates_png(self, tmp_path: Path) -> None:
        """loss_curve() must create a .png file at save_path."""
        save_path = tmp_path / "loss.png"
        MetricsPlotter.loss_curve(make_history(), save_path)
        assert save_path.exists()

    def test_file_is_not_empty(self, tmp_path: Path) -> None:
        """The generated PNG must not be empty."""
        save_path = tmp_path / "loss.png"
        MetricsPlotter.loss_curve(make_history(), save_path)
        assert save_path.stat().st_size > 0

    def test_creates_parent_dir_if_missing(self, tmp_path: Path) -> None:
        """loss_curve() must create the parent directory when it does not exist."""
        save_path = tmp_path / "subdir" / "loss.png"
        MetricsPlotter.loss_curve(make_history(), save_path)
        assert save_path.exists()


class TestConfusionMatrixPlot:
    def test_creates_png(self, tmp_path: Path) -> None:
        """confusion_matrix_plot() must create a .png file at save_path."""
        cm = [[3, 1], [0, 4]]
        save_path = tmp_path / "cm.png"
        MetricsPlotter.confusion_matrix_plot(cm, ["cat", "dog"], save_path)
        assert save_path.exists()

    def test_file_is_not_empty(self, tmp_path: Path) -> None:
        """The generated PNG must not be empty."""
        cm = [[3, 1], [0, 4]]
        save_path = tmp_path / "cm.png"
        MetricsPlotter.confusion_matrix_plot(cm, ["cat", "dog"], save_path)
        assert save_path.stat().st_size > 0

    def test_creates_parent_dir_if_missing(self, tmp_path: Path) -> None:
        """confusion_matrix_plot() must create the parent directory when needed."""
        cm = [[3, 1], [0, 4]]
        save_path = tmp_path / "subdir" / "cm.png"
        MetricsPlotter.confusion_matrix_plot(cm, ["cat", "dog"], save_path)
        assert save_path.exists()


class TestAccuracyCurve:
    def test_creates_png(self, tmp_path: Path) -> None:
        """accuracy_curve() must create a .png file at save_path."""
        save_path = tmp_path / "acc.png"
        MetricsPlotter.accuracy_curve(make_history(), save_path)
        assert save_path.exists()
        assert save_path.stat().st_size > 0


class TestConfusionMatrixNormalized:
    def test_creates_png(self, tmp_path: Path) -> None:
        """confusion_matrix_normalized() must create a non-empty .png."""
        cm = [[3, 1], [0, 4]]
        save_path = tmp_path / "cm_norm.png"
        MetricsPlotter.confusion_matrix_normalized(cm, ["cat", "dog"], save_path)
        assert save_path.exists()
        assert save_path.stat().st_size > 0

    def test_handles_zero_row(self, tmp_path: Path) -> None:
        """A row that sums to zero must not raise (divide-by-zero safe)."""
        cm = [[0, 0], [0, 4]]
        save_path = tmp_path / "cm_norm_zero.png"
        MetricsPlotter.confusion_matrix_normalized(cm, ["cat", "dog"], save_path)
        assert save_path.exists()


class TestROCCurve:
    def test_binary_creates_png(self, tmp_path: Path) -> None:
        """ROC curve for binary tasks must save a non-empty .png."""
        save_path = tmp_path / "roc.png"
        y_true = [0, 0, 1, 1]
        y_proba_full = [[0.9, 0.1], [0.6, 0.4], [0.3, 0.7], [0.2, 0.8]]
        ok = MetricsPlotter.roc_curve_plot(
            y_true, y_proba_full, ["neg", "pos"], save_path
        )
        assert ok is True
        assert save_path.exists()

    def test_single_class_returns_false(self, tmp_path: Path) -> None:
        """ROC is undefined for a single class — must return False and not write."""
        save_path = tmp_path / "roc_undef.png"
        ok = MetricsPlotter.roc_curve_plot(
            [0, 0, 0], [[1.0, 0.0]] * 3, ["a", "b"], save_path
        )
        assert ok is False
        assert not save_path.exists()

    def test_multiclass_creates_png(self, tmp_path: Path) -> None:
        """ROC curve for multiclass must save a non-empty .png."""
        save_path = tmp_path / "roc_mc.png"
        y_true = [0, 1, 2, 0, 1, 2]
        y_proba_full = [
            [0.7, 0.2, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.2, 0.7],
            [0.6, 0.3, 0.1],
            [0.3, 0.5, 0.2],
            [0.2, 0.2, 0.6],
        ]
        ok = MetricsPlotter.roc_curve_plot(
            y_true, y_proba_full, ["a", "b", "c"], save_path
        )
        assert ok is True
        assert save_path.exists()


class TestPrecisionRecallCurve:
    def test_binary_creates_png(self, tmp_path: Path) -> None:
        """Precision-recall curve for binary tasks must save a non-empty .png."""
        save_path = tmp_path / "pr.png"
        y_true = [0, 0, 1, 1]
        y_proba_full = [[0.9, 0.1], [0.6, 0.4], [0.3, 0.7], [0.2, 0.8]]
        ok = MetricsPlotter.precision_recall_curve_plot(
            y_true, y_proba_full, ["neg", "pos"], save_path
        )
        assert ok is True
        assert save_path.exists()

    def test_multiclass_creates_png(self, tmp_path: Path) -> None:
        """Precision-recall curve for multiclass must save a non-empty .png."""
        save_path = tmp_path / "pr_mc.png"
        y_true = [0, 1, 2, 0, 1, 2]
        y_proba_full = [
            [0.7, 0.2, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.2, 0.7],
            [0.6, 0.3, 0.1],
            [0.3, 0.5, 0.2],
            [0.2, 0.2, 0.6],
        ]
        ok = MetricsPlotter.precision_recall_curve_plot(
            y_true, y_proba_full, ["a", "b", "c"], save_path
        )
        assert ok is True
        assert save_path.exists()

    def test_single_class_returns_false(self, tmp_path: Path) -> None:
        """Precision-recall undefined for a single class — must return False."""
        save_path = tmp_path / "pr_undef.png"
        ok = MetricsPlotter.precision_recall_curve_plot(
            [1, 1, 1], [[0.0, 1.0]] * 3, ["a", "b"], save_path
        )
        assert ok is False
        assert not save_path.exists()

    def test_misshaped_probs_returns_false(self, tmp_path: Path) -> None:
        """proba.shape[0] != len(y_true) must return False without writing."""
        save_path = tmp_path / "pr_bad.png"
        ok = MetricsPlotter.precision_recall_curve_plot(
            [0, 1], [[0.5, 0.5]], ["a", "b"], save_path
        )
        assert ok is False
        assert not save_path.exists()


class TestROCCurveEdgeCases:
    def test_misshaped_probs_returns_false(self, tmp_path: Path) -> None:
        """ROC must refuse a proba array whose length disagrees with y_true."""
        save_path = tmp_path / "roc_bad.png"
        ok = MetricsPlotter.roc_curve_plot(
            [0, 1, 0], [[0.5, 0.5]], ["a", "b"], save_path
        )
        assert ok is False
        assert not save_path.exists()

    def test_multiclass_skips_classes_with_no_samples(self, tmp_path: Path) -> None:
        """A class that never appears in y_true must be skipped without erroring."""
        save_path = tmp_path / "roc_mc_skip.png"
        # 3-class probas but only classes 0 and 2 ever appear.
        y_true = [0, 2, 0, 2]
        y_proba_full = [
            [0.7, 0.1, 0.2],
            [0.1, 0.2, 0.7],
            [0.6, 0.2, 0.2],
            [0.2, 0.1, 0.7],
        ]
        ok = MetricsPlotter.roc_curve_plot(
            y_true, y_proba_full, ["a", "b", "c"], save_path
        )
        assert ok is True
        assert save_path.exists()
