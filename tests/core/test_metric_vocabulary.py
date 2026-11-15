"""One vocabulary decides which way a metric improves and where it lives (STAT-4).

The rule used to be a substring list that missed most error metrics outside the
built-in tasks (mape, nll, wer, fid...), so a custom task without declared
directions ranked them backwards; the GUI repeated the list by hand.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from visionforge.core import metric_vocabulary
from visionforge.core.replicates import metric_domain
from visionforge.core.significance import infer_direction

_TS = (
    Path(__file__).resolve().parents[2]
    / "frontend"
    / "src"
    / "lib"
    / "compare-metrics.ts"
)

LOWER = "lower"
HIGHER = "higher"

_LOWER_NAMES = [
    "test_mape",
    "val_smape",
    "msle",
    "rmsle",
    "nll",
    "val_bce",
    "huber_loss",
    "perplexity",
    "cer",
    "wer",
    "hd95",
    "fid",
    "eer",
    "fpr",
    "ece",
    "brier_score",
]
_HIGHER_NAMES = [
    "accuracy",
    "f1",
    "auroc",
    "map50",
    "miou",
    "dice",
    "r2",
    "answer_rate",
    "power",
]


@pytest.mark.parametrize("name", _LOWER_NAMES)
def test_error_metrics_outside_the_builtins_are_lower_is_better(name: str) -> None:
    assert infer_direction(name) == LOWER


@pytest.mark.parametrize("name", _HIGHER_NAMES)
def test_scores_stay_higher_is_better(name: str) -> None:
    assert infer_direction(name) == HIGHER


@pytest.mark.parametrize(
    "name", ["loss", "val_mae", "test_rmse", "mse", "error_rate", "top1_err", "l2dist"]
)
def test_the_old_substring_rule_still_holds(name: str) -> None:
    # Nothing that ranked as lower before turns into higher ("l2dist" is not
    # "distance"; it stays higher as it always was).
    expected = HIGHER if name == "l2dist" else LOWER
    assert infer_direction(name) == expected


@pytest.mark.parametrize("name", ["confidence", "piece_count", "career", "fidelity"])
def test_short_tokens_match_whole_words_only(name: str) -> None:
    # "fid" inside "confidence", "ece" inside "piece", "eer" inside "career".
    assert infer_direction(name) == HIGHER


@pytest.mark.parametrize("name", ["wer2", "val_fid50"])
def test_a_short_token_may_carry_a_number(name: str) -> None:
    assert infer_direction(name) == LOWER


def test_mape_is_floored_at_zero() -> None:
    assert metric_domain("test_mape") == (0.0, None)


@pytest.mark.parametrize("name", ["val_bce", "huber_loss", "fid", "test_wer"])
def test_non_negative_errors_are_floored_at_zero(name: str) -> None:
    assert metric_domain(name) == (0.0, None)


def test_a_gaussian_nll_is_not_floored() -> None:
    # A density's negative log-likelihood can go below zero.
    assert metric_domain("nll") == (None, None)


def _ts_list(source: str, name: str) -> list[str]:
    match = re.search(rf"const {name} = \[(.*?)\];", source, re.DOTALL)
    assert match is not None, name
    return re.findall(r'"([^"]+)"', match.group(1))


def test_the_gui_mirrors_the_vocabulary() -> None:
    source = _TS.read_text(encoding="utf-8")

    substrings = _ts_list(source, "LOWER_IS_BETTER_SUBSTRINGS")
    words = _ts_list(source, "LOWER_IS_BETTER_WORDS")

    assert substrings == list(metric_vocabulary.LOWER_IS_BETTER_SUBSTRINGS)
    assert words == list(metric_vocabulary.LOWER_IS_BETTER_WORDS)
