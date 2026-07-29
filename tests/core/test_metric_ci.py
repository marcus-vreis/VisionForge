"""Tests for the single-run bootstrap intervals (ADR-074).

The module trades sklearn calls for vectorized arithmetic, so the contract these
tests defend is *exactness*: every metric, on every averaging mode, must equal
what sklearn would return for the same resample. Anything else is a silent
statistical bug — an interval that looks plausible but describes a different
number than the report headlines.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from visionforge.core.metric_ci import (
    MetricCI,
    _auc_batch,
    _confusion_metrics,
    bootstrap_classification_cis,
)


def _binary_case(n: int = 400, seed: int = 0, coarse: bool = False):
    rng = np.random.default_rng(seed)
    true = rng.integers(0, 2, n)
    # A model that is right more often than not, so the metrics are not 0.5.
    flip = rng.random(n) < 0.25
    pred = np.where(flip, 1 - true, true)
    score = np.clip(true + rng.normal(0, 0.6, n), 0, 1)
    if coarse:
        score = np.round(score, 1)  # heavy ties, the case naive ranking breaks
    proba = np.stack([1.0 - score, score], axis=1)
    return true, pred, proba


def _multiclass_case(n: int = 400, k: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    true = rng.integers(0, k, n)
    flip = rng.random(n) < 0.3
    pred = np.where(flip, rng.integers(0, k, n), true)
    proba = rng.random((n, k)) + np.eye(k)[true] * 2.0
    proba /= proba.sum(axis=1, keepdims=True)
    return true, pred, proba


class TestVectorizedMetricsMatchSklearn:
    """The core exactness guard: vectorized == sklearn, resample by resample."""

    @pytest.mark.parametrize(
        ("n_classes", "average"),
        [(2, "binary"), (2, "macro"), (3, "macro"), (7, "macro")],
    )
    def test_confusion_metrics_match_sklearn(self, n_classes, average):
        rng = np.random.default_rng(7)
        n, rows = 300, 40
        true = rng.integers(0, n_classes, n)
        pred = rng.integers(0, n_classes, n)
        idx = rng.integers(0, n, (rows, n))

        acc, prec, rec, f1 = _confusion_metrics(true, pred, idx, n_classes, average)

        for row in range(rows):
            i = idx[row]
            assert acc[row] == pytest.approx(accuracy_score(true[i], pred[i]))
            assert prec[row] == pytest.approx(
                precision_score(true[i], pred[i], average=average, zero_division=0)
            )
            assert rec[row] == pytest.approx(
                recall_score(true[i], pred[i], average=average, zero_division=0)
            )
            assert f1[row] == pytest.approx(
                f1_score(true[i], pred[i], average=average, zero_division=0)
            )

    @pytest.mark.parametrize("coarse", [False, True])
    def test_auc_batch_matches_sklearn_including_ties(self, coarse):
        rng = np.random.default_rng(11)
        n, rows = 250, 40
        true = rng.integers(0, 2, n)
        score = rng.integers(0, 4, n).astype(float) if coarse else rng.random(n)
        _, positions = np.unique(score, return_inverse=True)
        positions = positions.astype(np.int64).ravel()
        idx = rng.integers(0, n, (rows, n))

        values, usable = _auc_batch(true, positions, int(positions.max()) + 1, idx)

        assert usable.all()  # both classes are plentiful here
        for row in range(rows):
            i = idx[row]
            assert values[row] == pytest.approx(roc_auc_score(true[i], score[i]))

    def test_macro_average_follows_sklearn_when_a_class_is_absent(self):
        """sklearn averages over labels present in the resample, not all classes."""
        true = np.array([0, 0, 1, 1])
        pred = np.array([0, 1, 1, 1])
        idx = np.array([[0, 1, 2, 3]])  # class 2 declared but never seen

        _, prec, _, _ = _confusion_metrics(true, pred, idx, 3, "macro")

        assert prec[0] == pytest.approx(
            precision_score(true, pred, average="macro", zero_division=0)
        )


class TestPointEstimates:
    def test_binary_point_estimates_equal_the_evaluator_metrics(self):
        true, pred, proba = _binary_case()

        cis = bootstrap_classification_cis(
            true, pred, task="binary", y_proba_full=proba, n_resamples=200
        )

        assert cis["accuracy"].value == pytest.approx(accuracy_score(true, pred))
        assert cis["f1"].value == pytest.approx(
            f1_score(true, pred, average="binary", zero_division=0)
        )
        assert cis["precision"].value == pytest.approx(
            precision_score(true, pred, average="binary", zero_division=0)
        )
        assert cis["recall"].value == pytest.approx(
            recall_score(true, pred, average="binary", zero_division=0)
        )
        assert cis["auc_roc"].value == pytest.approx(roc_auc_score(true, proba[:, 1]))

    def test_multiclass_point_estimates_equal_the_evaluator_metrics(self):
        true, pred, proba = _multiclass_case()

        cis = bootstrap_classification_cis(
            true, pred, task="multiclass", y_proba_full=proba, n_resamples=200
        )

        assert cis["accuracy"].value == pytest.approx(accuracy_score(true, pred))
        assert cis["f1"].value == pytest.approx(
            f1_score(true, pred, average="macro", zero_division=0)
        )
        assert cis["auc_roc"].value == pytest.approx(
            roc_auc_score(true, proba, multi_class="ovr", average="macro")
        )


class TestIntervals:
    def test_interval_brackets_the_point_estimate(self):
        true, pred, proba = _binary_case()

        cis = bootstrap_classification_cis(
            true, pred, task="binary", y_proba_full=proba
        )

        for ci in cis.values():
            assert ci.ci_low <= ci.value <= ci.ci_high, ci

    def test_more_samples_narrow_the_interval(self):
        narrow = bootstrap_classification_cis(
            *_binary_case(n=4000)[:2], task="binary", seed=3
        )["accuracy"]
        wide = bootstrap_classification_cis(
            *_binary_case(n=100)[:2], task="binary", seed=3
        )["accuracy"]

        assert (narrow.ci_high - narrow.ci_low) < (wide.ci_high - wide.ci_low)

    def test_confidence_level_widens_the_interval(self):
        true, pred, _ = _binary_case()
        tight = bootstrap_classification_cis(
            true, pred, task="binary", confidence=0.80
        )["accuracy"]
        loose = bootstrap_classification_cis(
            true, pred, task="binary", confidence=0.99
        )["accuracy"]

        assert (loose.ci_high - loose.ci_low) > (tight.ci_high - tight.ci_low)

    def test_same_seed_reproduces_the_interval(self):
        true, pred, proba = _binary_case()
        kwargs = {"task": "binary", "y_proba_full": proba, "n_resamples": 300}

        first = bootstrap_classification_cis(true, pred, seed=42, **kwargs)
        again = bootstrap_classification_cis(true, pred, seed=42, **kwargs)
        other = bootstrap_classification_cis(true, pred, seed=7, **kwargs)

        assert first["auc_roc"] == again["auc_roc"]
        assert first["auc_roc"].value == other["auc_roc"].value  # same split
        assert first["auc_roc"].ci_low != other["auc_roc"].ci_low  # different draws

    def test_resample_count_is_reported(self):
        true, pred, _ = _binary_case()

        ci = bootstrap_classification_cis(true, pred, task="binary", n_resamples=250)[
            "accuracy"
        ]

        assert ci.n_resamples == 250
        assert ci.n_samples == len(true)
        assert ci.confidence == 0.95


class TestGuards:
    def test_tiny_split_gets_no_interval_instead_of_a_fake_one(self):
        rng = np.random.default_rng(0)
        true = rng.integers(0, 2, 12)

        assert bootstrap_classification_cis(true, true, task="binary") == {}

    def test_single_class_split_still_reports_the_other_metrics(self):
        true = np.zeros(50, dtype=int)
        pred = np.zeros(50, dtype=int)
        proba = np.stack([np.ones(50), np.zeros(50)], axis=1)

        cis = bootstrap_classification_cis(
            true, pred, task="binary", y_proba_full=proba
        )

        assert "auc_roc" not in cis  # undefined without both classes
        assert cis["accuracy"].value == pytest.approx(1.0)

    def test_rare_class_drops_the_resamples_that_lose_it(self):
        """A class with one image cannot be in every resample — say so."""
        rng = np.random.default_rng(5)
        true = np.concatenate([rng.integers(0, 2, 99), [2]])
        pred = true.copy()
        proba = np.eye(3)[true] * 0.7 + 0.1

        cis = bootstrap_classification_cis(
            true, pred, task="multiclass", y_proba_full=proba, n_resamples=500
        )

        assert 0 < cis["auc_roc"].n_resamples < 500
        assert cis["accuracy"].n_resamples == 500  # accuracy is always defined

    def test_length_mismatch_is_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            bootstrap_classification_cis([0, 1, 0], [0, 1], task="binary")

    def test_proba_row_mismatch_is_rejected(self):
        true = np.zeros(30, dtype=int)
        with pytest.raises(ValueError, match="one row per sample"):
            bootstrap_classification_cis(
                true, true, task="binary", y_proba_full=np.zeros((10, 2))
            )

    def test_no_probabilities_means_no_auc_but_valid_metrics(self):
        true, pred, _ = _binary_case()

        cis = bootstrap_classification_cis(true, pred, task="binary")

        assert set(cis) == {"accuracy", "f1", "precision", "recall"}


class TestSerialization:
    def test_to_dict_is_json_serializable(self):
        true, pred, proba = _binary_case()

        ci = bootstrap_classification_cis(
            true, pred, task="binary", y_proba_full=proba, n_resamples=100
        )["accuracy"]
        payload = json.loads(json.dumps(ci.to_dict()))

        assert payload["metric"] == "accuracy"
        assert set(payload) == {
            "metric",
            "value",
            "ci_low",
            "ci_high",
            "confidence",
            "n_resamples",
            "n_samples",
        }

    def test_metric_ci_is_frozen(self):
        ci = MetricCI("accuracy", 0.9, 0.85, 0.95, 0.95, 1000, 400)

        with pytest.raises(AttributeError):
            ci.value = 0.5  # type: ignore[misc]
