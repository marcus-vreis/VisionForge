"""Target statistics of a regression dataset (ADR-235), on hand-computed values."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from PIL import Image

from visionforge.core.cancellation import CancellationToken
from visionforge.core.target_stats import (
    analyze_targets,
    csv_signature,
    histogram,
    pearson,
)

CSVS = {"train": "train.csv", "val": "val.csv", "test": "test.csv"}


def _dataset(root: Path) -> Path:
    """train: a = 1, 2, 3, 4, 100 and b = 2a, plus a row whose a is "x" and an
    image that does not exist. val: a = 3, 3. No test CSV."""
    (root / "images").mkdir(parents=True)
    for name in ("p1", "p2", "p3", "p4", "v1", "v2", "bad"):
        Image.new("RGB", (4, 4)).save(root / "images" / f"{name}.png")
    (root / "train.csv").write_text(
        "image,a,b\n"
        "p1.png,1,2\np2.png,2,4\np3.png,3,6\np4.png,4,8\n"
        "gone.png,100,200\n"
        "bad.png,x,5\n",
        encoding="utf-8",
    )
    (root / "val.csv").write_text(
        "image,a,b\nv1.png,3,6\nv2.png,3,6\n", encoding="utf-8"
    )
    return root


def _analyze(root: Path, **kw: object) -> dict:
    return analyze_targets(root, CSVS, targets=["a", "b"], **kw)  # type: ignore[arg-type]


def test_summary_outliers_and_shift(tmp_path: Path) -> None:
    report = _analyze(_dataset(tmp_path))
    a = report["targets"][0]
    train, val = a["splits"]["train"], a["splits"]["val"]
    assert (train["count"], train["bad"]) == (5, 1)
    assert train["mean"] == 22.0
    # Inclusive quartiles of 1, 2, 3, 4, 100: 2, 3, 4; IQR 2; fences -1 and 7.
    assert (train["q1"], train["median"], train["q3"]) == (2.0, 3.0, 4.0)
    assert a["fences"] == [-1.0, 7.0]
    assert train["outlier_count"] == 1
    assert train["outliers"] == [{"line": 6, "image": "gone.png", "value": 100.0}]
    # Sample std of 1, 2, 3, 4, 100 is sqrt(7610 / 4); val sits 19 below the mean.
    assert val["shift_sd"] == pytest.approx(-19 / math.sqrt(7610 / 4))
    assert train["shift_sd"] == 0.0
    assert "test" not in a["splits"]
    assert report["splits"]["test"]["present"] is False


def test_histogram_bins_are_shared_by_the_splits(tmp_path: Path) -> None:
    a = _analyze(_dataset(tmp_path))["targets"][0]
    assert a["range"] == [1.0, 100.0]
    assert a["splits"]["train"]["histogram"][0] == 4
    assert a["splits"]["train"]["histogram"][-1] == 1
    assert a["splits"]["val"]["histogram"][0] == 2
    assert sum(a["splits"]["val"]["histogram"]) == 2


def test_missing_images_bad_cells_and_correlation(tmp_path: Path) -> None:
    report = _analyze(_dataset(tmp_path))
    train = report["splits"]["train"]
    assert train["rows"] == 6
    assert train["missing_images"] == 1
    assert train["missing_examples"] == [{"line": 6, "image": "gone.png"}]
    assert train["bad_examples"] == [{"line": 7, "target": "a", "cell": "x"}]
    # b = 2a on every row where both are numbers.
    assert report["correlations"] == [{"a": "a", "b": "b", "r": pytest.approx(1.0)}]


def test_a_missing_column_is_named(tmp_path: Path) -> None:
    _dataset(tmp_path)
    report = analyze_targets(tmp_path, CSVS, targets=["a", "c"])
    assert report["splits"]["train"]["missing_columns"] == ["c"]
    assert report["targets"][1]["splits"]["train"]["count"] == 0


def test_a_csv_outside_the_root_is_refused(tmp_path: Path) -> None:
    _dataset(tmp_path / "ds")
    with pytest.raises(ValueError, match="outside"):
        analyze_targets(tmp_path / "ds", {"train": "../x.csv"}, targets=["a"])


def test_cancelled_before_the_first_csv(tmp_path: Path) -> None:
    token = CancellationToken()
    token.cancel()
    report = _analyze(_dataset(tmp_path), token=token)
    assert report["cancelled"] is True
    assert report["splits"] == {}


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([0.0, 1.0, 2.0, 3.0], [1, 1, 2]),  # 3.0 sits on the top edge: last bin
        ([5.0, 5.0], [0, 0, 2]),  # no spread: everything in the last bin
    ],
)
def test_histogram_edges(values: list[float], expected: list[int]) -> None:
    lo, hi = min(values), max(values)
    assert histogram(values, lo, hi, bins=3) == expected


def test_pearson_needs_three_pairs_and_a_spread() -> None:
    assert pearson([(1, 2), (2, 4)]) is None
    assert pearson([(1, 5), (2, 5), (3, 5)]) is None
    assert pearson([(1, 3), (2, 2), (3, 1)]) == pytest.approx(-1.0)


def test_the_signature_follows_the_csvs(tmp_path: Path) -> None:
    _dataset(tmp_path)
    before = csv_signature(tmp_path, CSVS)
    (tmp_path / "val.csv").write_text("image,a,b\nv1.png,4,8\n", encoding="utf-8")
    assert csv_signature(tmp_path, CSVS) != before
