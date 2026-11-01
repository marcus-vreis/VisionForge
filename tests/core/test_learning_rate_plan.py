"""What a job will train at, as the Review shows it (ADR-158)."""

from __future__ import annotations

from visionforge.core.learning_rate import (
    collapse_evidence,
    comparison_trial_rate,
    grid_learning_rate_plan,
    grid_trial_rate,
)
from visionforge.gui.api.schemas import LearningRatePlanRequest
from visionforge.gui.api.system_service import learning_rate_plan

LR_AXIS = "training.learning_rate"


class TestComparisonTrialRate:
    def test_fixed_keeps_the_configured_rate(self) -> None:
        trial = comparison_trial_rate("vgg16", "adam", 0.01, "fixed")
        assert trial.learning_rate == 0.01

    def test_suggested_ignores_the_configured_rate(self) -> None:
        trial = comparison_trial_rate("vgg16", "adam", 0.01, "suggested")
        assert trial.learning_rate == 1e-4

    def test_flags_only_the_trio_measured_to_fail(self) -> None:
        fixed = comparison_trial_rate("vgg16", "adam", 1e-3, "fixed")
        suggested = comparison_trial_rate("vgg16", "adam", 1e-3, "suggested")
        assert fixed.collapse_prone is True
        assert fixed.evidence == collapse_evidence("vgg16")
        assert suggested.collapse_prone is False
        assert suggested.evidence is None


class TestGridTrialRate:
    def test_rate_has_no_float_noise(self) -> None:
        assert (
            grid_trial_rate("vgg16", "adam", 3, "suggested_multiplier").learning_rate
            == 3e-4
        )

    def test_absolute_has_no_multiplier(self) -> None:
        trial = grid_trial_rate("vgg16", "adam", 0.01, "absolute")
        assert (trial.learning_rate, trial.multiplier) == (0.01, None)

    def test_multiplier_is_kept_beside_the_rate(self) -> None:
        trial = grid_trial_rate("resnet50", "adam", 0.3, "suggested_multiplier")
        assert (trial.learning_rate, trial.multiplier) == (3e-4, 0.3)


class TestGridPlan:
    def test_factor_above_one_on_a_fragile_model_is_flagged(self) -> None:
        plan = grid_learning_rate_plan(
            {"model.name": ["vgg16", "resnet50"], LR_AXIS: [1, 3]},
            "resnet18",
            "adam",
            1e-3,
            "suggested_multiplier",
        )
        flagged = [(t.architecture, t.learning_rate) for t in plan if t.collapse_prone]
        # vgg16 at 3x is 3e-4, above the 1e-4 it is safe at; resnet50 is never flagged.
        assert flagged == [("vgg16", 3e-4)]

    def test_absent_axes_fall_back_to_the_base_values(self) -> None:
        plan = grid_learning_rate_plan({}, "resnet18", "sgd", 0.02)
        assert [(t.architecture, t.optimizer, t.learning_rate) for t in plan] == [
            ("resnet18", "sgd", 0.02)
        ]

    def test_axes_that_do_not_touch_the_rate_repeat_nothing(self) -> None:
        plan = grid_learning_rate_plan(
            {"training.batch_size": [8, 16], "model.name": ["resnet18"]},
            "resnet18",
            "adam",
            1e-3,
        )
        assert len(plan) == 1

    def test_values_still_being_typed_are_skipped(self) -> None:
        plan = grid_learning_rate_plan(
            {LR_AXIS: ["", "abc", 0, True, 0.01]}, "resnet18", "adam", 1e-3
        )
        assert [t.learning_rate for t in plan] == [0.01]


def _request(**fields: object) -> LearningRatePlanRequest:
    base: dict[str, object] = {
        "block": "classification",
        "model_name": "vgg16",
        "optimizer": "adam",
        "learning_rate": 1e-3,
    }
    return LearningRatePlanRequest.model_validate({**base, **fields})


class TestLearningRatePlanService:
    def test_single_run_is_one_flagged_row_with_its_evidence(self) -> None:
        [trial] = learning_rate_plan(_request()).trials
        assert trial.collapse_prone is True
        assert trial.collapse_evidence is not None
        assert trial.collapse_evidence.measured_on == "vgg16"
        assert trial.collapse_evidence.accuracy == 0.25

    def test_comparison_lists_each_model(self) -> None:
        response = learning_rate_plan(
            _request(
                block="model_comparison",
                model_names=["vgg16", "resnet50"],
                lr_policy="suggested",
            )
        )
        assert [(t.architecture, t.learning_rate) for t in response.trials] == [
            ("vgg16", 1e-4),
            ("resnet50", 1e-3),
        ]

    def test_grid_uses_the_multiplier(self) -> None:
        response = learning_rate_plan(
            _request(
                block="grid_search",
                hyperparameters={"model.name": ["vgg16"], LR_AXIS: [3]},
                lr_mode="suggested_multiplier",
            )
        )
        assert response.trials[0].learning_rate == 3e-4
        assert response.trials[0].multiplier == 3

    def test_random_search_claims_nothing(self) -> None:
        assert learning_rate_plan(_request(block="random_search")).trials == []
