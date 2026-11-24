"""`TaskCvRequest.select_on` (ADR-242): which epoch scores each task K-fold fold."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from visionforge.gui.api.schemas import TaskCvRequest


class TestSelectOn:
    def test_defaults_to_best_so_an_older_client_keeps_its_behaviour(self) -> None:
        assert TaskCvRequest(config={}).select_on == "best"

    @pytest.mark.parametrize("rule", ["best", "last"])
    def test_accepts_the_two_rules(self, rule: str) -> None:
        req = TaskCvRequest.model_validate({"config": {}, "select_on": rule})

        assert req.select_on == rule

    def test_rejects_any_other_value(self) -> None:
        with pytest.raises(ValidationError, match="select_on"):
            TaskCvRequest.model_validate({"config": {}, "select_on": "middle"})


class TestLastWithEarlyStopping:
    """`last` is refused when the task config stops early (ADR-242 addendum)."""

    def test_last_with_patience_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="early_stopping_patience"):
            TaskCvRequest.model_validate(
                {
                    "config": {"training": {"early_stopping_patience": 2}},
                    "select_on": "last",
                }
            )

    @pytest.mark.parametrize(
        ("rule", "patience"), [("last", 0), ("best", 2), ("best", 0)]
    )
    def test_other_combinations_are_accepted(self, rule: str, patience: int) -> None:
        req = TaskCvRequest.model_validate(
            {
                "config": {"training": {"early_stopping_patience": patience}},
                "select_on": rule,
            }
        )

        assert req.select_on == rule


class TestLastWithPlateau:
    """`last` is refused when the task config uses the plateau scheduler."""

    def test_last_with_plateau_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="plateau"):
            TaskCvRequest.model_validate(
                {
                    "config": {"training": {"scheduler": {"kind": "plateau"}}},
                    "select_on": "last",
                }
            )

    @pytest.mark.parametrize(
        ("rule", "kind"), [("last", "cosine"), ("last", "none"), ("best", "plateau")]
    )
    def test_other_combinations_are_accepted(self, rule: str, kind: str) -> None:
        req = TaskCvRequest.model_validate(
            {
                "config": {"training": {"scheduler": {"kind": kind}}},
                "select_on": rule,
            }
        )

        assert req.select_on == rule
