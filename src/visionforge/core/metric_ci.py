"""Bootstrap confidence intervals for a single run's test metrics (ADR-074).

ADR-056 answers "how uncertain is this number?" by training N times under N
seeds. That is the stronger answer and it costs N trainings. This module answers
a narrower question from **one** training: given this fixed model, how much of
the reported metric is an accident of *which images landed in the test split*?
Resampling the test set with replacement gives a percentile interval for that,
so a run reports ``0.873 [0.841, 0.902]`` instead of a bare ``0.873``.

The two intervals are not interchangeable and the report should not present them
as if they were: this one holds the model fixed and varies the evaluation
sample, so it captures test-set sampling noise only — not the run-to-run
variance from initialization, data order and nondeterministic kernels that
replicates measure. A tight interval here says nothing about whether retraining
would land in the same place.

Everything is a pure function over per-sample arrays — no torch, no config — so
it is testable against sklearn and reusable by any task that keeps its
predictions.

Implementation note: the metrics are recomputed with vectorized confusion-matrix
and rank arithmetic rather than by calling sklearn once per resample. That is
not premature optimization — sklearn's per-call overhead dominates (~5 ms
regardless of split size), which put 1000 resamples at ~5 s per run and would
have made the interval something the researcher has to opt into. Vectorized it
is ~0.02 s on a 500-image split, so it can simply always be there. The tests
pin every path against sklearn to keep the shortcut honest.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

# Keeps the (resamples x samples) index matrices to a few million entries so a
# large test split chunks instead of allocating gigabytes.
_MAX_CELLS = 2_000_000

# Below this the interval is arithmetic, not evidence: a handful of images
# cannot describe a population, and a narrow interval would be an artifact of
# resampling the same few values.
_MIN_SAMPLES = 20


@dataclass(frozen=True)
class MetricCI:
    """Percentile bootstrap interval for one metric on one evaluation split.

    Deliberately not ``significance.BootstrapCI``: that one describes the *mean
    over seeds* and names its point estimate ``mean``. Here the point estimate
    is the metric measured on the real split, and the resampling unit is the
    sample, not the seed. Sharing a shape would blur which uncertainty a number
    refers to.
    """

    metric: str
    value: float  # measured on the actual split, not an average of resamples
    ci_low: float
    ci_high: float
    confidence: float
    n_resamples: int  # resamples in which this metric was defined
    n_samples: int

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form for the run report."""
        return asdict(self)


def bootstrap_classification_cis(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    task: Literal["binary", "multiclass"],
    y_proba_full: Sequence[Sequence[float]] | None = None,
    confidence: float = 0.95,
    n_resamples: int = 1000,
    seed: int = 0,
) -> dict[str, MetricCI]:
    """Percentile CIs for accuracy, F1, precision, recall and AUC-ROC.

    Averaging follows the point estimate the ``Evaluator`` reports: ``binary``
    scores the positive class, ``multiclass`` macro-averages. ``auc_roc`` is
    included only when ``y_proba_full`` is given and the split supports it.

    Returns an empty mapping for a split too small to resample honestly
    (under 20 samples), so callers get "no interval" rather than a fabricated one.
    """
    true = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    if true.size != pred.size:
        raise ValueError(
            f"y_true and y_pred must have the same length ({true.size} != {pred.size})."
        )
    n = int(true.size)
    if n < _MIN_SAMPLES or n_resamples < 2:
        return {}

    proba = (
        np.asarray(y_proba_full, dtype=float)
        if y_proba_full is not None and len(y_proba_full) > 0
        else None
    )
    if proba is not None and proba.shape[0] != n:
        raise ValueError(
            f"y_proba_full must have one row per sample "
            f"({proba.shape[0]} rows for {n} samples)."
        )

    n_classes = (
        proba.shape[1] if proba is not None else int(max(true.max(), pred.max())) + 1
    )
    average: Literal["binary", "macro"] = "binary" if task == "binary" else "macro"

    rng = np.random.default_rng(seed)
    collected: dict[str, list[np.ndarray]] = {
        "accuracy": [],
        "f1": [],
        "precision": [],
        "recall": [],
    }
    auc_values: list[np.ndarray] = []
    auc_ready = _prepare_auc(true, proba, task, n_classes)

    drawn = 0
    while drawn < n_resamples:
        chunk = min(_chunk_size(n), n_resamples - drawn)
        idx = rng.integers(0, n, size=(chunk, n))
        acc, prec, rec, f1 = _confusion_metrics(true, pred, idx, n_classes, average)
        collected["accuracy"].append(acc)
        collected["precision"].append(prec)
        collected["recall"].append(rec)
        collected["f1"].append(f1)
        if auc_ready is not None:
            values, usable = auc_ready(idx)
            auc_values.append(values[usable])
        drawn += chunk

    point, auc_point = _point_estimates(
        true, pred, proba, n_classes, average, auc_ready
    )
    out: dict[str, MetricCI] = {}
    for name, parts in collected.items():
        out[name] = _interval(name, point[name], np.concatenate(parts), confidence, n)
    if auc_values and auc_point is not None:
        merged = np.concatenate(auc_values)
        # Every resample that lost a class was dropped, so a split with a
        # one-image class can leave too few to quantile.
        if merged.size >= 2:
            out["auc_roc"] = _interval("auc_roc", auc_point, merged, confidence, n)
    return out


def _interval(
    metric: str,
    value: float,
    samples: np.ndarray,
    confidence: float,
    n_samples: int,
) -> MetricCI:
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(samples, [tail, 1.0 - tail])
    return MetricCI(
        metric=metric,
        value=float(value),
        ci_low=float(low),
        ci_high=float(high),
        confidence=confidence,
        n_resamples=int(samples.size),
        n_samples=n_samples,
    )


def _chunk_size(n: int) -> int:
    return max(1, min(200, _MAX_CELLS // max(n, 1)))


def _confusion_metrics(
    true: np.ndarray,
    pred: np.ndarray,
    idx: np.ndarray,
    n_classes: int,
    average: Literal["binary", "macro"],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Accuracy/precision/recall/F1 for a batch of resamples, sklearn-compatible.

    One ``bincount`` over the flattened (resample, true, pred) cell index builds
    every confusion matrix at once; zero denominators score 0, matching
    ``zero_division=0``.
    """
    rows, n = idx.shape
    cells = true[idx] * n_classes + pred[idx]
    offsets = (np.arange(rows) * n_classes * n_classes)[:, None]
    counts = np.bincount(
        (cells + offsets).ravel(), minlength=rows * n_classes * n_classes
    )
    cm = counts.reshape(rows, n_classes, n_classes).astype(float)

    tp = np.einsum("rii->ri", cm)
    support = cm.sum(axis=2)  # true instances per class
    predicted = cm.sum(axis=1)  # predicted instances per class

    precision = np.where(predicted > 0, tp / np.where(predicted > 0, predicted, 1), 0.0)
    recall = np.where(support > 0, tp / np.where(support > 0, support, 1), 0.0)
    denom = precision + recall
    f1 = np.where(
        denom > 0, 2 * precision * recall / np.where(denom > 0, denom, 1), 0.0
    )
    accuracy = tp.sum(axis=1) / n

    if average == "binary":
        pos = 1 if n_classes > 1 else 0
        return accuracy, precision[:, pos], recall[:, pos], f1[:, pos]

    # sklearn macro-averages over the labels present in the resample (as a true
    # or a predicted label), not over every declared class.
    present = (support > 0) | (predicted > 0)
    n_present = np.maximum(present.sum(axis=1), 1)
    macro = lambda values: (values * present).sum(axis=1) / n_present  # noqa: E731
    return accuracy, macro(precision), macro(recall), macro(f1)


def _prepare_auc(
    true: np.ndarray,
    proba: np.ndarray | None,
    task: str,
    n_classes: int,
) -> Any:
    """Build a resample→AUC callable, or ``None`` when AUC does not apply.

    Scores are mapped once to their position among the sorted distinct values,
    which is what makes the tie handling exact: a bootstrap resample duplicates
    rows by construction, so ties are the common case, not an edge case, and
    ranking without them silently inflates the AUC.
    """
    if proba is None or proba.shape[1] < 2:
        return None
    if len(np.unique(true)) < 2:
        return None

    if task == "binary":
        columns = [(np.asarray(true == 1, dtype=np.int64), proba[:, 1])]
    else:
        columns = [
            (np.asarray(true == c, dtype=np.int64), proba[:, c])
            for c in range(n_classes)
        ]

    prepared = []
    for labels, scores in columns:
        _, positions = np.unique(scores, return_inverse=True)
        prepared.append(
            (labels, positions.astype(np.int64).ravel(), int(positions.max()) + 1)
        )

    def compute(idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        per_class = []
        usable = np.ones(idx.shape[0], dtype=bool)
        for labels, positions, n_values in prepared:
            values, ok = _auc_batch(labels, positions, n_values, idx)
            per_class.append(np.nan_to_num(values))
            # A resample missing this class cannot produce the macro AUC the
            # point estimate reports, so the whole resample is dropped rather
            # than averaged over a different set of classes.
            usable &= ok
        return np.mean(per_class, axis=0), usable

    return compute


def _auc_batch(
    labels: np.ndarray,
    positions: np.ndarray,
    n_values: int,
    idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Tie-aware ROC AUC per resample, plus which resamples had both classes."""
    rows = idx.shape[0]
    cells = positions[idx] * 2 + labels[idx]
    offsets = (np.arange(rows) * n_values * 2)[:, None]
    counts = np.bincount((cells + offsets).ravel(), minlength=rows * n_values * 2)
    grid = counts.reshape(rows, n_values, 2).astype(float)
    negatives, positives = grid[:, :, 0], grid[:, :, 1]

    n_neg = negatives.sum(axis=1)
    n_pos = positives.sum(axis=1)
    usable = (n_pos > 0) & (n_neg > 0)
    # Mann-Whitney form: negatives strictly below each score value, with ties
    # splitting the credit — exactly what roc_auc_score does.
    below = np.cumsum(negatives, axis=1) - negatives
    concordant = (positives * below).sum(axis=1) + 0.5 * (positives * negatives).sum(
        axis=1
    )
    denom = np.where(usable, n_pos * n_neg, 1.0)
    return np.where(usable, concordant / denom, np.nan), usable


def _point_estimates(
    true: np.ndarray,
    pred: np.ndarray,
    proba: np.ndarray | None,
    n_classes: int,
    average: Literal["binary", "macro"],
    auc_ready: Any,
) -> tuple[dict[str, float], float | None]:
    """The metrics on the real split, computed by the same code as the resamples.

    Using one code path for both means the interval can never straddle a
    different number than the one the report headlines. AUC comes back separately
    because it is the only one that can be undefined.
    """
    identity = np.arange(true.size, dtype=np.int64)[None, :]
    acc, prec, rec, f1 = _confusion_metrics(true, pred, identity, n_classes, average)
    values = {
        "accuracy": float(acc[0]),
        "precision": float(prec[0]),
        "recall": float(rec[0]),
        "f1": float(f1[0]),
    }
    auc_point: float | None = None
    if auc_ready is not None:
        auc, usable = auc_ready(identity)
        if bool(usable[0]):
            auc_point = float(auc[0])
    return values, auc_point


__all__ = ["MetricCI", "bootstrap_classification_cis"]
