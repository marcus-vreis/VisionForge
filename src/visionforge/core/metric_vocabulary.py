"""Which way a metric improves and which values it can take, by its name.

The fallback for a metric whose task declared no direction (a custom task, a
sweep over a researcher's own score): ``infer_direction`` and ``metric_domain``
both read this vocabulary, and the GUI's ``compare-metrics.ts`` mirrors the two
direction lists (``tests/core/test_metric_vocabulary.py`` compares them).

Long tokens match anywhere in the name ("huber_loss", "hausdorff_distance");
short ones only as a whole word of it, split on anything that is not a letter or
digit, optionally followed by a number ("val_fid50", "wer2"). Substring matching
for a three-letter token would read "fid" in "confidence" or "wer" in "power".
The first seven substrings are the original rule, kept as substrings so no name
that ranked lower-is-better before changes side.
"""

from __future__ import annotations

import re

LOWER_IS_BETTER_SUBSTRINGS: tuple[str, ...] = (
    "loss",
    "mae",
    "mse",
    "rmse",
    "error",
    "err",
    "distance",
    "perplexity",
    "huber",
    "brier",
)

LOWER_IS_BETTER_WORDS: tuple[str, ...] = (
    "msle",
    "rmsle",
    "mape",
    "smape",
    "nll",
    "bce",
    "wer",
    "cer",
    "hd95",
    "fid",
    "eer",
    "fpr",
    "ece",
)

# Lower-is-better names that cannot go below zero by construction. A negative
# log-likelihood is left out on purpose: a density's NLL can be negative.
ABSOLUTE_WORDS: frozenset[str] = frozenset(
    {
        "mae",
        "mse",
        "rmse",
        "msle",
        "rmsle",
        "mape",
        "smape",
        "loss",
        "bce",
        "huber",
        "perplexity",
        "wer",
        "cer",
        "hd95",
        "fid",
        "brier",
        "eer",
        "fpr",
        "ece",
    }
)

_DIGITS = "0123456789"
_WORD = re.compile(rf"^({'|'.join(LOWER_IS_BETTER_WORDS)})\d*$")


def metric_words(metric: str) -> list[str]:
    """The lowercase words of a metric name ("val_MAPE" -> ["val", "mape"])."""
    return re.split(r"[^a-z0-9]+", metric.lower())


def is_lower_is_better(metric: str) -> bool:
    """Whether the name reads as an error, a loss or a rate of mistakes."""
    lowered = metric.lower()
    if any(token in lowered for token in LOWER_IS_BETTER_SUBSTRINGS):
        return True
    return any(_WORD.match(word) for word in metric_words(metric))


def is_absolute_error(metric: str) -> bool:
    """A lower-is-better name whose values are non-negative by construction."""
    words = metric_words(metric)
    if any(w in ABSOLUTE_WORDS or w.rstrip(_DIGITS) in ABSOLUTE_WORDS for w in words):
        return True
    return "error" in words and ("absolute" in words or "squared" in words)


__all__ = [
    "ABSOLUTE_WORDS",
    "LOWER_IS_BETTER_SUBSTRINGS",
    "LOWER_IS_BETTER_WORDS",
    "is_absolute_error",
    "is_lower_is_better",
    "metric_words",
]
