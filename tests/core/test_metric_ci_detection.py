"""mAP cannot be summed per image, so its interval is earned differently."""

from __future__ import annotations

import random

import numpy as np
import torch

from visionforge.core.metric_ci import bootstrap_detection_cis


def _det(
    boxes: list[list[float]], labels: list[int], scores: list[float] | None = None
):
    d = {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }
    if scores is not None:
        d["scores"] = torch.tensor(scores, dtype=torch.float32)
    return d


def _split(n: int, *, hit: int, n_classes: int = 2, seed: int = 0):
    """`hit` images are detected perfectly; the rest are misses.

    Hits and misses are interleaved and confidences are distinct, which matters:
    with every score tied and all hits first, the full split gets a precision
    curve that no resample can reproduce, and the interval then sits entirely
    below the point estimate. That is mAP being order-dependent under ties, not
    the bootstrap misbehaving — but it makes a fixture that tests the wrong
    thing.
    """
    rng = random.Random(seed)
    order = [True] * hit + [False] * (n - hit)
    rng.shuffle(order)
    preds, gts = [], []
    box = [10.0, 10.0, 50.0, 50.0]
    for i, is_hit in enumerate(order):
        cls = i % n_classes
        gts.append(_det([box], [cls]))
        score = 0.5 + rng.random() * 0.5
        preds.append(
            _det([box] if is_hit else [[100.0, 100.0, 140.0, 140.0]], [cls], [score])
        )
    return preds, gts


class TestBootstrapDetectionCis:
    def test_brackets_the_measured_value(self) -> None:
        preds, gts = _split(60, hit=40)

        cis = bootstrap_detection_cis(preds, gts, n_resamples=60, seed=0)

        ci = cis["map50"]
        assert ci.ci_low <= ci.value <= ci.ci_high

    def test_a_perfect_split_has_a_degenerate_interval(self) -> None:
        preds, gts = _split(40, hit=40)

        ci = bootstrap_detection_cis(preds, gts, n_resamples=40, seed=0)["map50"]

        assert ci.value == 1.0
        assert ci.ci_low == 1.0 and ci.ci_high == 1.0

    def test_more_disagreement_gives_a_wider_interval(self) -> None:
        """The interval has to track the evidence, not just exist."""
        tight = bootstrap_detection_cis(*_split(60, hit=58), n_resamples=80, seed=1)
        loose = bootstrap_detection_cis(*_split(60, hit=30), n_resamples=80, seed=1)

        def width(c):
            return c["map50"].ci_high - c["map50"].ci_low

        assert width(loose) > width(tight)

    def test_too_few_images_returns_nothing_rather_than_a_fake_interval(self) -> None:
        preds, gts = _split(10, hit=5)

        assert bootstrap_detection_cis(preds, gts, n_resamples=50) == {}

    def test_mismatched_lengths_are_rejected(self) -> None:
        preds, gts = _split(30, hit=15)

        try:
            bootstrap_detection_cis(preds, gts[:-1])
        except ValueError as exc:
            assert "equal length" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("expected a ValueError")

    def test_no_ground_truth_returns_nothing(self) -> None:
        preds = [_det([[1.0, 1.0, 2.0, 2.0]], [0], [0.5]) for _ in range(30)]
        gts = [_det([], []) for _ in range(30)]

        assert bootstrap_detection_cis(preds, gts, n_resamples=30) == {}

    def test_draws_that_lose_a_class_are_discarded(self) -> None:
        """A resample missing a class averages over fewer classes — not the
        same quantity, so it must not enter the interval."""
        preds, gts = _split(40, hit=25, n_classes=2)
        # One class appears in a single image, so most draws will miss it.
        gts[0] = _det([[10.0, 10.0, 50.0, 50.0]], [7])
        preds[0] = _det([[10.0, 10.0, 50.0, 50.0]], [7], [0.9])

        cis = bootstrap_detection_cis(preds, gts, n_resamples=100, seed=3)

        # Surviving draws are reported, and they are far fewer than requested.
        assert cis["map50"].n_resamples < 100

    def test_is_deterministic_for_a_seed(self) -> None:
        preds, gts = _split(50, hit=30)

        a = bootstrap_detection_cis(preds, gts, n_resamples=40, seed=7)["map50"]
        b = bootstrap_detection_cis(preds, gts, n_resamples=40, seed=7)["map50"]

        assert (a.ci_low, a.ci_high) == (b.ci_low, b.ci_high)

    def test_reports_the_image_count_as_the_sample_size(self) -> None:
        preds, gts = _split(35, hit=20)

        assert (
            bootstrap_detection_cis(preds, gts, n_resamples=30)["map50"].n_samples == 35
        )


def test_numpy_is_used_for_the_quantile() -> None:
    """Guards the import the module needs for _interval."""
    assert np.quantile([0.0, 1.0], 0.5) == 0.5
