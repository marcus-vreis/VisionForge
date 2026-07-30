"""Bootstrap intervals for regression, anomaly and segmentation (ADR-076).

Each entry point is pinned against the accumulator that actually produces the
number a run reports. That is the whole contract: an interval computed by a
second implementation is worse than no interval if the two disagree, because it
would bracket a value the report never shows.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.metrics import f1_score

from visionforge.core.anomaly_trainer import compute_auroc
from visionforge.core.metric_ci import (
    bootstrap_anomaly_cis,
    bootstrap_regression_cis,
    bootstrap_segmentation_cis,
)
from visionforge.core.regression_trainer import _MetricAccumulator
from visionforge.core.segmentation_trainer import SegmentationMetricAccumulator


class TestRegressionCis:
    @pytest.mark.parametrize("targets", [1, 3])
    def test_point_estimates_match_the_streaming_accumulator(
        self, targets: int
    ) -> None:
        rng = np.random.default_rng(3)
        true = rng.normal(5.0, 2.0, (200, targets))
        pred = true + rng.normal(0, 0.7, (200, targets))

        acc = _MetricAccumulator()
        acc.update(torch.tensor(pred), torch.tensor(true))
        mse, rmse, mae, r2 = acc.compute()

        cis = bootstrap_regression_cis(true, pred, n_resamples=100)

        assert cis["mse"].value == pytest.approx(mse)
        assert cis["rmse"].value == pytest.approx(rmse)
        assert cis["mae"].value == pytest.approx(mae)
        assert cis["r2"].value == pytest.approx(r2)

    def test_accepts_a_flat_single_target_sequence(self) -> None:
        rng = np.random.default_rng(4)
        true = rng.normal(0, 1, 60)

        cis = bootstrap_regression_cis(true, true + 0.1, n_resamples=50)

        assert cis["mae"].value == pytest.approx(0.1)
        assert cis["mae"].n_samples == 60

    def test_interval_brackets_every_point_estimate(self) -> None:
        rng = np.random.default_rng(5)
        true = rng.normal(0, 3, (300, 1))
        pred = true + rng.normal(0, 1, (300, 1))

        for ci in bootstrap_regression_cis(true, pred).values():
            assert ci.ci_low <= ci.value <= ci.ci_high, ci

    def test_more_samples_narrow_the_interval(self) -> None:
        rng = np.random.default_rng(11)

        def spread(n: int) -> float:
            true = rng.normal(0, 2, (n, 1))
            ci = bootstrap_regression_cis(true, true + rng.normal(0, 1, (n, 1)), seed=2)
            return ci["mae"].ci_high - ci["mae"].ci_low

        assert spread(2000) < spread(60)

    def test_shape_mismatch_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="same shape"):
            bootstrap_regression_cis(np.zeros((30, 2)), np.zeros((30, 3)))

    def test_constant_target_reports_r2_zero_like_the_accumulator(self) -> None:
        """No variance to explain: the accumulator returns 0.0, not 1.0."""
        true = np.full((40, 1), 7.0)

        cis = bootstrap_regression_cis(true, true, n_resamples=50)

        assert cis["r2"].value == 0.0
        assert cis["mse"].value == 0.0

    def test_tiny_split_gets_no_interval(self) -> None:
        assert bootstrap_regression_cis(np.zeros((5, 1)), np.zeros((5, 1))) == {}


class TestAnomalyCis:
    @staticmethod
    def _split(seed: int = 6) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        labels = np.concatenate([np.zeros(60, int), np.ones(40, int)])
        scores = np.concatenate([rng.normal(0.3, 0.1, 60), rng.normal(0.7, 0.1, 40)])
        return labels, scores

    def test_point_estimates_match_the_trainer_metrics(self) -> None:
        labels, scores = self._split()
        threshold = 0.5

        cis = bootstrap_anomaly_cis(labels, scores, threshold, n_resamples=100)

        expected_f1 = f1_score(
            labels, (scores >= threshold).astype(int), zero_division=0
        )
        expected_auroc = compute_auroc(torch.tensor(scores), torch.tensor(labels))
        assert cis["image_f1"].value == pytest.approx(expected_f1)
        assert cis["auroc"].value == pytest.approx(expected_auroc)

    def test_threshold_is_an_input_not_recomputed_per_resample(self) -> None:
        """It comes from the normal-train distribution, so it is part of the model."""
        labels, scores = self._split(seed=7)

        strict = bootstrap_anomaly_cis(labels, scores, 0.9, n_resamples=80)
        loose = bootstrap_anomaly_cis(labels, scores, 0.1, n_resamples=80)

        assert strict["image_f1"].value != loose["image_f1"].value

    def test_interval_brackets_the_point_estimate(self) -> None:
        labels, scores = self._split(seed=9)

        for ci in bootstrap_anomaly_cis(labels, scores, 0.5).values():
            assert ci.ci_low <= ci.value <= ci.ci_high, ci

    def test_single_class_split_still_reports_f1(self) -> None:
        labels = np.zeros(40, int)

        cis = bootstrap_anomaly_cis(labels, np.full(40, 0.1), 0.5, n_resamples=50)

        assert "auroc" not in cis  # undefined without both classes
        assert "image_f1" in cis

    def test_length_mismatch_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            bootstrap_anomaly_cis([0, 1, 0], [0.1, 0.2], 0.5)


class TestSegmentationCis:
    @staticmethod
    def _matrices(n: int, k: int, seed: int) -> np.ndarray:
        """Diagonal-heavy per-image matrices — a trained model, not a random one."""
        rng = np.random.default_rng(seed)
        base = rng.integers(0, 30, (n, k, k)).astype(float)
        for i in range(k):
            base[:, i, i] += rng.integers(200, 400, n)
        return base

    @staticmethod
    def _accumulator_metrics(matrices: np.ndarray) -> tuple[float, float, float]:
        k = matrices.shape[1]
        acc = SegmentationMetricAccumulator(num_classes=k, ignore_index=255)
        acc._cm = torch.tensor(matrices.sum(axis=0), dtype=torch.long)
        return acc.compute()

    def test_point_estimates_match_the_streaming_accumulator(self) -> None:
        matrices = self._matrices(50, 4, seed=8)
        miou, dice, pixel_acc = self._accumulator_metrics(matrices)

        cis = bootstrap_segmentation_cis(matrices, n_resamples=100)

        assert cis["miou"].value == pytest.approx(miou)
        assert cis["dice"].value == pytest.approx(dice)
        assert cis["pixel_acc"].value == pytest.approx(pixel_acc)

    def test_averages_over_classes_present_as_truth_or_prediction(self) -> None:
        """A class that is only ever predicted still counts, as the accumulator does."""
        one = np.array([[80.0, 0.0, 10.0], [0.0, 90.0, 5.0], [0.0, 0.0, 0.0]])
        matrices = np.repeat(one[None, :, :], 30, axis=0)

        cis = bootstrap_segmentation_cis(matrices, n_resamples=50)

        assert cis["miou"].value == pytest.approx(
            self._accumulator_metrics(matrices)[0]
        )

    def test_resampling_unit_is_the_image(self) -> None:
        """Identical images must give a zero-width interval.

        With pixels as the unit, resampling would manufacture spread the
        evidence does not contain. This is the guard against that.
        """
        one = np.array([[100.0, 5.0], [3.0, 120.0]])
        matrices = np.repeat(one[None, :, :], 40, axis=0)

        ci = bootstrap_segmentation_cis(matrices, n_resamples=80)["miou"]

        assert ci.ci_low == pytest.approx(ci.ci_high)

    def test_interval_brackets_the_point_estimate(self) -> None:
        matrices = self._matrices(80, 3, seed=12)

        for ci in bootstrap_segmentation_cis(matrices).values():
            assert ci.ci_low <= ci.value <= ci.ci_high, ci

    def test_non_square_input_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="square"):
            bootstrap_segmentation_cis(np.zeros((30, 2, 3)))

    def test_tiny_split_gets_no_interval(self) -> None:
        assert bootstrap_segmentation_cis(np.ones((5, 3, 3))) == {}
