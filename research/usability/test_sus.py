from __future__ import annotations

import pytest
from sus import quiz_gain, summarize, sus_score


class TestSusScore:
    def test_neutral_answers_score_fifty(self) -> None:
        assert sus_score([3] * 10) == 50.0

    def test_the_best_possible_answers_score_one_hundred(self) -> None:
        assert sus_score([5, 1] * 5) == 100.0

    def test_the_worst_possible_answers_score_zero(self) -> None:
        assert sus_score([1, 5] * 5) == 0.0

    def test_a_worked_example(self) -> None:
        # odd: 4,5,4,4,3 -> 3+4+3+3+2 = 15; even: 2,1,2,1,2 -> 3+4+3+4+3 = 17.
        assert sus_score([4, 2, 5, 1, 4, 2, 4, 1, 3, 2]) == 80.0

    @pytest.mark.parametrize("answers", [[3] * 9, [3] * 11])
    def test_needs_exactly_ten_answers(self, answers: list[int]) -> None:
        with pytest.raises(ValueError, match="10 items"):
            sus_score(answers)

    @pytest.mark.parametrize("bad", [0, 6])
    def test_answers_are_one_to_five(self, bad: int) -> None:
        with pytest.raises(ValueError, match="1..5"):
            sus_score([3] * 9 + [bad])


class TestSummarize:
    def test_over_participants(self) -> None:
        s = summarize([80.0, 70.0, 90.0])
        assert s["mean"] == 80.0
        assert s["sd"] == pytest.approx(10.0)
        assert (s["min"], s["max"]) == (70.0, 90.0)

    def test_a_single_participant_has_no_spread(self) -> None:
        assert summarize([72.5])["sd"] == 0.0


class TestQuizGain:
    def test_paired_by_position(self) -> None:
        # gains: +2, 0, -1 -> mean 1/3.
        result = quiz_gain([1, 3, 5], [3, 3, 4])
        assert result["mean_gain"] == pytest.approx(1 / 3)
        assert (result["improved"], result["unchanged"], result["worse"]) == (1, 1, 1)

    @pytest.mark.parametrize(("pre", "post"), [([1, 2], [3]), ([], [])])
    def test_lists_must_pair(self, pre: list[int], post: list[int]) -> None:
        with pytest.raises(ValueError, match="pair one-to-one"):
            quiz_gain(pre, post)
