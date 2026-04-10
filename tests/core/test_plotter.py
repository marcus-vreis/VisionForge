from pathlib import Path

from visionforge.core.plotter import MetricsPlotter
from visionforge.core.trainer import EpochResult


def make_history(n: int = 5) -> list[EpochResult]:
    return [
        EpochResult(
            epoch=i, train_loss=1.0 / i, val_loss=1.2 / i, val_accuracy=0.5 + i * 0.05
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
