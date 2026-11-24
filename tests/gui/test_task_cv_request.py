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
