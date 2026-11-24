"""Whether a dataset's splits share the same class mix (ADR-232).

A validation or test split whose class proportions differ from training measures
the model on a different problem: a rare class over-represented in test drags the
metric down, one missing from validation is never checked. This module reads the
per-class counts the explorer already has (no image is opened) and says how far
the splits are from a stratified division.

Three numbers, from the most concrete to the most summarized:

* per class, the gap in percentage points between its largest and smallest share
  across the splits, and the splits where it is missing altogether;
* per split, the imbalance ratio (largest class count over smallest);
* Cramér's V of the split-by-class table: 0 when every split has the same mix,
  1 when the split determines the class. It is an effect size, not a test: with
  thousands of images any difference is "significant", and what matters is how
  large it is.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

# A gap up to this many percentage points is what a random or stratified split
# of a real dataset leaves (rounding of small classes); up to the second one the
# splits are close; past it a class's share clearly moves between splits.
STRATIFIED_GAP_PP = 2.0
CLOSE_GAP_PP = 5.0

Verdict = Literal["stratified", "close", "uneven", "missing_class", "one_split"]


def split_balance(
    per_class: Mapping[str, Mapping[str, int]], splits: Sequence[str]
) -> dict[str, Any]:
    """How the class mix differs between ``splits`` (each ``per_class[split]``).

    Only splits that hold at least one labeled image count. Shares are taken over
    the labels of a split, so for detection (an image per class it contains) they
    sum to 100 as for classification.
    """
    used = [s for s in splits if sum(per_class.get(s, {}).values()) > 0]
    classes = sorted({c for s in used for c, n in per_class[s].items() if n > 0})
    table = {s: [per_class[s].get(c, 0) for c in classes] for s in used}
    rows = [_class_row(c, i, table) for i, c in enumerate(classes)]
    rows.sort(key=lambda r: (-r["gap_pp"], r["name"]))
    max_gap = max((r["gap_pp"] for r in rows), default=0.0)
    return {
        "splits": used,
        "classes": rows,
        "max_gap_pp": max_gap,
        "cramers_v": cramers_v([table[s] for s in used]),
        "imbalance": {s: _imbalance(table[s]) for s in used},
        "verdict": _verdict(len(used), rows, max_gap),
        "thresholds_pp": [STRATIFIED_GAP_PP, CLOSE_GAP_PP],
    }


def cramers_v(table: Sequence[Sequence[int]]) -> float | None:
    """Cramér's V of a contingency table (rows: splits, columns: classes).

    None when it is undefined: fewer than two rows or columns with counts.
    """
    rows = [list(r) for r in table if sum(r) > 0]
    if len(rows) < 2:
        return None
    columns = [j for j in range(len(rows[0])) if sum(r[j] for r in rows) > 0]
    if len(columns) < 2:
        return None
    n = sum(sum(r) for r in rows)
    row_sums = [sum(r) for r in rows]
    col_sums = [sum(r[j] for r in rows) for j in columns]
    chi2 = 0.0
    for i, row in enumerate(rows):
        for k, j in enumerate(columns):
            expected = row_sums[i] * col_sums[k] / n
            chi2 += (row[j] - expected) ** 2 / expected
    return math.sqrt(chi2 / (n * (min(len(rows), len(columns)) - 1)))


def _class_row(name: str, i: int, table: Mapping[str, list[int]]) -> dict[str, Any]:
    """One class: its share (%) in each split, the gap, and where it is missing."""
    shares = {s: 100.0 * counts[i] / sum(counts) for s, counts in table.items()}
    present = list(shares.values())
    return {
        "name": name,
        "counts": {s: counts[i] for s, counts in table.items()},
        "shares": shares,
        "gap_pp": (max(present) - min(present)) if len(present) > 1 else 0.0,
        "missing_in": [s for s, counts in table.items() if counts[i] == 0],
    }


def _imbalance(counts: list[int]) -> float | None:
    """Largest class over smallest, among the classes the split holds."""
    held = [n for n in counts if n > 0]
    return max(held) / min(held) if len(held) > 1 else None


def _verdict(n_splits: int, rows: list[dict[str, Any]], max_gap: float) -> Verdict:
    """A one-word reading of the numbers, for the screen's headline."""
    if n_splits < 2:
        return "one_split"
    if any(r["missing_in"] for r in rows):
        return "missing_class"
    if max_gap <= STRATIFIED_GAP_PP:
        return "stratified"
    return "close" if max_gap <= CLOSE_GAP_PP else "uneven"
