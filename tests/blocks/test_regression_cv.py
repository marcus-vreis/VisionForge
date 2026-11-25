from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest
from PIL import Image

from visionforge.blocks.regression_cv import (
    FoldResult,
    aggregate_folds,
    run_regression_cross_validation,
)
from visionforge.utils.regression_config import RegressionConfig

_N_ROWS = 6


def _make_dataset(tmp_path: Path) -> Path:
    base = tmp_path / "ds"
    images = base / "images"
    images.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(_N_ROWS):
        Image.new("RGB", (32, 32), color=(i * 10, 20, 30)).save(images / f"img{i}.png")
        rows.append({"image": f"img{i}.png", "target": float(i)})
    with (base / "train.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "target"])
        writer.writeheader()
        writer.writerows(rows)
    return base


def _config(tmp_path: Path) -> RegressionConfig:
    base = _make_dataset(tmp_path)
    return RegressionConfig.model_validate(
        {
            "name": "reg_cv",
            "model": {"name": "resnet18", "num_targets": 1, "pretrained": False},
            "data": {
                "base_dir": str(base),
                "target_columns": ["target"],
                "transforms": {"image_size": 32},
            },
            "training": {"epochs": 1, "batch_size": 2},
            "output": {"models_dir": str(tmp_path / "models")},
        }
    )


class TestRegressionCrossValidation:
    def test_two_fold_cv_runs_and_aggregates(self, tmp_path: Path) -> None:
        report = run_regression_cross_validation(
            _config(tmp_path), n_folds=2, shuffle=True, seed=42
        )

        assert report.n_folds == 2
        assert report.metric == "r2"
        assert len(report.folds) == 2
        assert all(f.status == "success" for f in report.folds), [
            f.error for f in report.folds
        ]

        # the folds partition all rows with no leakage
        assert sum(f.val_size for f in report.folds) == _N_ROWS
        assert all(f.train_size + f.val_size == _N_ROWS for f in report.folds)

        # each fold names the run it trained, so History can fold it under the
        # K-fold's group (ADR-123)
        for fold in report.folds:
            assert fold.run_dir is not None
            assert (Path(fold.run_dir) / "run.json").is_file()

        # every regression metric is aggregated to mean ± std
        for name in ("r2", "rmse", "mae", "mse"):
            assert name in report.aggregate
            assert {"mean", "std"} <= report.aggregate[name].keys()

    def test_rejects_too_few_folds(self, tmp_path: Path) -> None:
        config = _config(tmp_path)
        with pytest.raises(ValueError, match="n_folds"):
            run_regression_cross_validation(config, n_folds=1)

    def test_rejects_more_folds_than_rows(self, tmp_path: Path) -> None:
        config = _config(tmp_path)
        with pytest.raises(ValueError, match="exceed"):
            run_regression_cross_validation(config, n_folds=_N_ROWS + 1)


class TestAggregateFoldsWithNonFiniteFolds:
    """A fold whose metric is NaN or inf is not a measurement (ADR-061)."""

    @staticmethod
    def _fold(index: int, r2: float) -> FoldResult:
        return FoldResult(
            fold=index, status="success", train_size=4, val_size=2, metrics={"r2": r2}
        )

    def test_two_nan_folds_stay_out_and_are_counted(self) -> None:
        folds = [
            self._fold(0, math.nan),
            self._fold(1, 0.5),
            self._fold(2, math.nan),
            self._fold(3, 0.7),
        ]

        aggregate = aggregate_folds(folds, ("r2",))

        r2 = aggregate["r2"]
        assert r2["n"] == 2
        assert r2["n_non_finite"] == 2
        assert r2["mean"] == pytest.approx(0.6)
        assert r2["std"] == pytest.approx(0.1414213562)

    def test_every_fold_non_finite_keeps_the_metric_with_no_statistics(self) -> None:
        folds = [self._fold(0, math.nan), self._fold(1, math.inf)]

        r2 = aggregate_folds(folds, ("r2",))["r2"]

        assert r2["n"] == 0
        assert r2["n_non_finite"] == 2
        assert r2["mean"] is None
        assert r2["std"] is None

    def test_finite_folds_report_zero_non_finite(self) -> None:
        folds = [self._fold(0, 0.5), self._fold(1, 0.7)]

        r2 = aggregate_folds(folds, ("r2",))["r2"]

        assert r2["n"] == 2
        assert r2["n_non_finite"] == 0
