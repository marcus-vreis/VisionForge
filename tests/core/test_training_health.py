"""A number that looks like a result but is not one has to say so.

Found by training the real grid (ADR-099): VGG16 and AlexNet with Adam at the
default 1e-3 predict a single class for every image. On four classes that reads
as accuracy 0.25; on two it reads as exactly 0.50 — a value easy to mistake for
"trained a little". Nothing in the log or the report said otherwise.
"""

from __future__ import annotations

from visionforge.core.training_health import (
    collapsed_predictions,
    constant_predictions,
    stagnant_loss,
    summarize,
)


class TestCollapsedPredictions:
    def test_one_class_for_everything_is_reported(self) -> None:
        warning = collapsed_predictions([1] * 400, n_classes=2)

        assert warning is not None
        assert warning.code == "collapsed_predictions"
        assert "mesma classe" in warning.message

    def test_a_model_that_distinguishes_is_not_flagged(self) -> None:
        assert collapsed_predictions([0, 1, 1, 0, 1], n_classes=2) is None

    def test_a_single_wrong_prediction_still_counts_as_distinguishing(self) -> None:
        """The check is about collapse, not about accuracy."""
        assert collapsed_predictions([0] * 399 + [1], n_classes=2) is None

    def test_nothing_to_say_about_an_empty_run(self) -> None:
        assert collapsed_predictions([], n_classes=2) is None

    def test_a_one_class_problem_cannot_collapse(self) -> None:
        assert collapsed_predictions([0] * 10, n_classes=1) is None


class TestStagnantLoss:
    def test_a_loss_that_never_moved_is_reported(self) -> None:
        # The real VGG16+Adam numbers: 1.446 -> 1.393 over three epochs.
        warning = stagnant_loss([1.446, 1.402, 1.393])

        assert warning is not None
        assert warning.code == "stagnant_loss"

    def test_a_learning_run_is_not_flagged(self) -> None:
        # The real VGG16+SGD numbers from the same grid.
        assert stagnant_loss([1.214, 0.812, 0.615]) is None

    def test_one_epoch_says_nothing_either_way(self) -> None:
        assert stagnant_loss([1.4]) is None

    def test_it_reads_the_best_epoch_not_the_last(self) -> None:
        """A loss that dropped and then bounced still learned something."""
        assert stagnant_loss([1.4, 0.3, 1.39]) is None


class TestConstantPredictions:
    def test_a_regressor_predicting_the_mean_is_reported(self) -> None:
        warning = constant_predictions([31.4] * 50)

        assert warning is not None
        assert warning.code == "constant_predictions"

    def test_real_spread_is_not_flagged(self) -> None:
        assert constant_predictions([10.0, 25.0, 60.0, 41.5]) is None


class TestSummarize:
    def test_it_keeps_only_what_happened(self) -> None:
        got = summarize(
            [collapsed_predictions([1] * 10, 2), None, stagnant_loss([1.0, 0.999])]
        )

        assert [w["code"] for w in got] == ["collapsed_predictions", "stagnant_loss"]

    def test_a_healthy_run_says_nothing(self) -> None:
        assert summarize([None, None]) == []
