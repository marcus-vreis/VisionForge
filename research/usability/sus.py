"""Scoring for the first-tester evaluation: SUS (Brooke, 1996) and the concept quiz.

SUS has ten items answered 1 (strongly disagree) to 5 (strongly agree). Odd items
are worded positively and contribute ``answer - 1``; even items negatively and
contribute ``5 - answer``. The sum (0-40) times 2.5 gives 0-100. It is not a
percentage, and it is only interpretable against other SUS scores.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence


def sus_score(answers: Sequence[int]) -> float:
    """One participant's SUS score, 0-100.

    Raises:
        ValueError: not exactly ten answers, or an answer outside 1..5.
    """
    if len(answers) != 10:
        raise ValueError(f"SUS has 10 items, got {len(answers)} answers.")
    if any(a not in range(1, 6) for a in answers):
        raise ValueError(f"Every answer must be 1..5, got {list(answers)}.")
    total = sum((a - 1) if i % 2 == 0 else (5 - a) for i, a in enumerate(answers))
    return total * 2.5


def summarize(scores: Sequence[float]) -> dict[str, float]:
    """Mean, SD and range over participants; SD is 0 for a single participant."""
    return {
        "n": float(len(scores)),
        "mean": statistics.fmean(scores),
        "sd": statistics.stdev(scores) if len(scores) > 1 else 0.0,
        "min": min(scores),
        "max": max(scores),
    }


def quiz_gain(pre: Sequence[int], post: Sequence[int]) -> dict[str, float]:
    """Paired change in quiz score (post - pre) over participants, in the same order.

    Raises:
        ValueError: the two lists differ in length or are empty.
    """
    if len(pre) != len(post) or not pre:
        raise ValueError(
            f"pre and post must pair one-to-one, got {len(pre)} and {len(post)}."
        )
    gains = [b - a for a, b in zip(pre, post, strict=True)]
    return {
        "n": float(len(gains)),
        "mean_gain": statistics.fmean(gains),
        "improved": float(sum(g > 0 for g in gains)),
        "unchanged": float(sum(g == 0 for g in gains)),
        "worse": float(sum(g < 0 for g in gains)),
    }
