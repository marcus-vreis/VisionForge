from __future__ import annotations

import math

import pytest
from analysis import (
    noise_to_effect,
    paired_differences,
    per_seed,
    reversal_rate,
    seeds_for_power,
)

REPORT = {
    "variants": {
        "baseline": {"per_seed": {"0": 0.80, "1": 0.84, "2": 0.78}},
        "no_aug": {"per_seed": {"0": 0.82, "1": 0.83, "2": 0.81, "3": 0.90}},
    }
}


class TestPairing:
    def test_seeds_come_back_as_integers(self) -> None:
        assert per_seed(REPORT, "baseline") == {0: 0.80, 1: 0.84, 2: 0.78}

    def test_only_seeds_both_variants_finished_are_paired(self) -> None:
        diffs = paired_differences(REPORT, "no_aug")
        assert diffs == pytest.approx([0.02, -0.01, 0.03])


class TestReversalRate:
    def test_single_seeds_that_disagree_with_the_mean(self) -> None:
        # mean +0.5; one of four single seeds points the other way.
        assert reversal_rate([1, 1, 1, -1], k=1) == 0.25

    def test_all_seeds_never_reverse(self) -> None:
        assert reversal_rate([1, 1, 1, -1], k=4) == 0.0

    def test_pairs_enumerated_by_hand(self) -> None:
        # mean +0.75; of the six pairs only (-1, -1) points down.
        assert reversal_rate([2, -1, -1, 3], k=2) == pytest.approx(1 / 6)

    def test_a_tied_subset_counts_as_a_reversal(self) -> None:
        # (1, -1) averages to zero: it would not have shown the effect.
        assert reversal_rate([1, -1, 2], k=2) == pytest.approx(1 / 3)

    def test_a_negative_effect_is_reversed_by_positive_subsets(self) -> None:
        assert reversal_rate([-1, -1, -1, 1], k=1) == 0.25

    def test_large_sets_are_sampled_reproducibly(self) -> None:
        diffs = [0.01 * (i % 7) - 0.02 for i in range(20)]
        first = reversal_rate(diffs, k=10, draws=500, seed=3)
        assert first == reversal_rate(diffs, k=10, draws=500, seed=3)
        assert 0.0 <= first <= 1.0

    @pytest.mark.parametrize("k", [0, 5])
    def test_k_out_of_range(self, k: int) -> None:
        with pytest.raises(ValueError, match="k must be"):
            reversal_rate([1, 2, 3, 4], k=k)

    def test_no_overall_effect(self) -> None:
        with pytest.raises(ValueError, match="zero"):
            reversal_rate([1, -1], k=1)


class TestSeedsForPower:
    def test_effect_equal_to_its_spread(self) -> None:
        # mean 1, sd 1 -> (1.95996 + 0.84162)^2 = 7.85 -> 8 seeds.
        assert seeds_for_power([0.0, 1.0, 2.0]) == 8

    def test_never_below_two(self) -> None:
        assert seeds_for_power([1.0, 1.001, 0.999]) == 2

    def test_no_effect_needs_infinitely_many(self) -> None:
        assert seeds_for_power([1.0, -1.0]) is None


class TestNoiseToEffect:
    def test_noise_twice_the_effect(self) -> None:
        # baseline sd 0.02 (0.78, 0.80, 0.82); mean effect 0.01.
        assert noise_to_effect([0.78, 0.80, 0.82], [0.01, 0.01, 0.01]) == pytest.approx(
            2.0
        )

    def test_zero_effect_is_infinite(self) -> None:
        assert math.isinf(noise_to_effect([0.78, 0.80], [0.01, -0.01]))
