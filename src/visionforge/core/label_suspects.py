"""Images whose label a trained classifier disputes (ADR-237).

Confident learning (Northcutt, Jiang and Chuang, "Confident Learning:
Estimating Uncertainty in Dataset Labels", JAIR 2021) in its simplest form:

* the threshold of class ``j`` is the mean probability the model gives ``j`` on
  the images labeled ``j`` (its average self-confidence), capped at
  ``THRESHOLD_CEILING``. The cap departs from the paper: a network trained to
  saturation gives its classes a mean self-confidence of 0.999, and an image the
  model calls ``j`` with 0.998 would then never count toward ``j``;
* an image labeled ``i`` counts toward class ``j`` when ``p_j >= t_j``; among
  the classes that pass, the one with the largest probability wins;
* the image is a suspect when that class is not ``i``. Suspects are ranked by
  how little probability the model leaves to the given label.

The counts of (given, confident) form the confident joint; its largest
off-diagonal cells are the pairs of classes the model takes for each other.

The probabilities should come from images the model did not train on (held-out
or out-of-fold); on its own training images a model agrees with the labels it
memorized, and the screen says so.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

MAX_SUSPECTS = 500
# Above this, a threshold only measures how saturated the softmax is.
THRESHOLD_CEILING = 0.95
TOP_PAIRS = 10


def class_thresholds(
    probs: np.ndarray, labels: np.ndarray, n_classes: int
) -> np.ndarray:
    """Mean self-confidence per class, capped at ``THRESHOLD_CEILING``.

    ``inf`` for a class with no image: nothing can be confidently that class.
    """
    out = np.full(n_classes, np.inf)
    for j in range(n_classes):
        mine = labels == j
        if mine.any():
            out[j] = min(float(probs[mine, j].mean()), THRESHOLD_CEILING)
    return out


def confident_labels(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Per image, the most probable class among those over their threshold; -1 if none."""
    passing = np.where(probs >= thresholds[None, :], probs, -np.inf)
    best = passing.argmax(axis=1)
    return np.where(np.isfinite(passing.max(axis=1)), best, -1)


def find_suspects(
    probs: np.ndarray, labels: np.ndarray, class_names: Sequence[str]
) -> dict[str, Any]:
    """Suspects, the confident joint and its largest off-diagonal pairs.

    ``probs`` is [N, C] (rows summing to 1); ``labels`` is [N] of ints in 0..C-1.
    """
    n_classes = len(class_names)
    thresholds = class_thresholds(probs, labels, n_classes)
    confident = confident_labels(probs, thresholds)
    joint = np.zeros((n_classes, n_classes), dtype=int)
    counted = confident >= 0
    np.add.at(joint, (labels[counted], confident[counted]), 1)
    flagged = np.flatnonzero(counted & (confident != labels))
    given_p = probs[flagged, labels[flagged]]
    order = flagged[np.argsort(given_p, kind="stable")]
    pairs = [
        (int(joint[i, j]), i, j)
        for i in range(n_classes)
        for j in range(n_classes)
        if i != j and joint[i, j] > 0
    ]
    pairs.sort(key=lambda p: (-p[0], p[1], p[2]))
    return {
        "thresholds": [None if np.isinf(t) else float(t) for t in thresholds],
        "suspect_count": int(len(order)),
        "suspects": [
            {
                "index": int(k),
                "given": class_names[labels[k]],
                "suspected": class_names[confident[k]],
                "p_given": float(probs[k, labels[k]]),
                "p_suspected": float(probs[k, confident[k]]),
            }
            for k in order[:MAX_SUSPECTS]
        ],
        "joint": joint.tolist(),
        "pairs": [
            {"given": class_names[i], "taken_for": class_names[j], "images": n}
            for n, i, j in pairs[:TOP_PAIRS]
        ],
    }


def read_folder_predictions(
    csv_path: Path, split_dir: Path, class_names: Sequence[str]
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Paths, probabilities and labels from a batch-prediction CSV over a split.

    The label of an image is its class folder (``<split>/<class>/<file>``);
    an image outside a known class folder is skipped. Paths come back relative
    to ``split_dir``.
    """
    index = {name: i for i, name in enumerate(class_names)}
    columns = [f"prob_class_{i}" for i in range(len(class_names))]
    paths: list[str] = []
    rows: list[list[float]] = []
    labels: list[int] = []
    root = split_dir.resolve()
    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        # A binary model writes one sigmoid probability: p(class 1).
        binary = "prob_positive" in (reader.fieldnames or [])
        for row in reader:
            path = Path(row["filename"]).resolve()
            try:
                rel = path.relative_to(root)
            except ValueError:
                continue
            if len(rel.parts) < 2 or rel.parts[0] not in index:
                continue
            paths.append(rel.as_posix())
            if binary:
                positive = float(row["prob_positive"])
                rows.append([1.0 - positive, positive])
            else:
                rows.append([float(row[c]) for c in columns])
            labels.append(index[rel.parts[0]])
    probs = np.array(rows, dtype=float).reshape(-1, len(class_names))
    return paths, probs, np.array(labels, dtype=int)
