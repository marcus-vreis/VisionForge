"""Whether the splits share the class mix (ADR-232); every number by hand."""

from __future__ import annotations

import pytest

from visionforge.core.split_balance import cramers_v, split_balance

SPLITS = ("train", "val", "test")


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        ([[5, 5], [5, 5]], 0.0),  # same mix
        ([[10, 0], [0, 10]], 1.0),  # the split decides the class
        # n=80, every expected count 20: chi2 = 4 * 10**2 / 20 = 20; sqrt(20/80).
        ([[30, 10], [10, 30]], 0.5),
    ],
)
def test_cramers_v_by_hand(table: list[list[int]], expected: float) -> None:
    assert cramers_v(table) == pytest.approx(expected)


def test_cramers_v_is_undefined_with_one_split_or_one_class() -> None:
    assert cramers_v([[3, 4]]) is None
    assert cramers_v([[3, 0], [5, 0]]) is None
    assert cramers_v([[3, 4], [0, 0]]) is None


def test_a_stratified_division() -> None:
    per_class = {
        "train": {"cat": 80, "dog": 20},
        "val": {"cat": 8, "dog": 2},
        "test": {"cat": 16, "dog": 4},
    }

    report = split_balance(per_class, SPLITS)

    assert report["verdict"] == "stratified"
    assert report["max_gap_pp"] == pytest.approx(0.0)
    assert report["cramers_v"] == pytest.approx(0.0)
    assert report["imbalance"] == {"train": 4.0, "val": 4.0, "test": 4.0}


def test_gaps_are_percentage_points_and_sorted_largest_first() -> None:
    # cat: 70% in train, 60% in val -> 10 pp; dog is the complement, also 10 pp.
    per_class = {"train": {"cat": 70, "dog": 30}, "val": {"cat": 6, "dog": 4}}

    report = split_balance(per_class, SPLITS)

    assert report["splits"] == ["train", "val"]
    assert [r["name"] for r in report["classes"]] == ["cat", "dog"]
    cat = report["classes"][0]
    assert cat["shares"] == pytest.approx({"train": 70.0, "val": 60.0})
    assert cat["gap_pp"] == pytest.approx(10.0)
    assert report["max_gap_pp"] == pytest.approx(10.0)
    assert report["verdict"] == "uneven"


@pytest.mark.parametrize(
    ("val_a", "verdict"), [(48, "stratified"), (46, "close"), (44, "uneven")]
)
def test_the_verdict_follows_the_thresholds(val_a: int, verdict: str) -> None:
    # train 50/50 and val val_a/(100 - val_a): gaps of 2, 4 and 6 points.
    per_class = {"train": {"a": 50, "b": 50}, "val": {"a": val_a, "b": 100 - val_a}}

    report = split_balance(per_class, SPLITS)

    assert report["verdict"] == verdict


def test_a_class_missing_from_a_split_is_named() -> None:
    per_class = {
        "train": {"cat": 50, "dog": 45, "fox": 5},
        "val": {"cat": 10, "dog": 10, "fox": 0},
    }

    report = split_balance(per_class, SPLITS)

    fox = next(r for r in report["classes"] if r["name"] == "fox")
    assert fox["missing_in"] == ["val"]
    assert report["verdict"] == "missing_class"
    assert report["imbalance"] == {"train": 10.0, "val": 1.0}


def test_one_split_or_an_empty_split_is_no_comparison() -> None:
    report = split_balance({"train": {"cat": 3, "dog": 1}, "val": {}}, SPLITS)

    assert report["splits"] == ["train"]
    assert report["verdict"] == "one_split"
    assert report["cramers_v"] is None
    assert report["classes"][0]["gap_pp"] == 0.0
