"""The validation confusion of one epoch, for the live "Results" tab (ADR-161).

Built from the predictions ``_eval_epoch`` already made, so it costs O(N)
integer counts and no forward pass. It travels in the ``epoch_end`` event only;
``run.json`` keeps the test-set confusion matrix, as before.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

# A full matrix up to this many classes: 30×30 integers per epoch. Above it the
# grid is unreadable in the sheet and heavy in the stream, so only the most
# frequent confusions are sent.
FULL_MATRIX_MAX_CLASSES = 30
TOP_CONFUSIONS = 10


def live_confusion(
    labels: Sequence[int],
    preds: Sequence[int],
    class_names: Sequence[str],
) -> dict[str, Any] | None:
    """The epoch's confusion as ``{"classes", "matrix"}`` or ``{"classes", "top"}``.

    ``matrix[i][j]`` counts validation images of class ``i`` predicted as ``j``.
    ``top`` lists ``[true, predicted, count]`` for the most frequent mistakes,
    most frequent first. ``None`` when there is nothing to count or a label falls
    outside the class list (a mismatch is not worth a wrong picture).
    """
    n = len(class_names)
    if n == 0 or not labels or len(labels) != len(preds):
        return None
    if any(not 0 <= v < n for v in (*labels, *preds)):
        return None
    classes = list(class_names)
    if n <= FULL_MATRIX_MAX_CLASSES:
        matrix = [[0] * n for _ in range(n)]
        for true, pred in zip(labels, preds, strict=True):
            matrix[true][pred] += 1
        return {"classes": classes, "matrix": matrix}
    mistakes = Counter(
        (true, pred) for true, pred in zip(labels, preds, strict=True) if true != pred
    )
    # Ties go to the lower class pair, so the list is the same on every machine.
    ranked = sorted(mistakes.items(), key=lambda item: (-item[1], item[0]))
    top = [[true, pred, count] for (true, pred), count in ranked[:TOP_CONFUSIONS]]
    return {"classes": classes, "top": top}
