"""Half a restored training state is worse than none: it looks continued."""

from __future__ import annotations

from pathlib import Path

import torch

from visionforge.core.resume import (
    RESUME_FILENAME,
    ResumeState,
    can_resume,
    clear_resume_state,
    load_resume_state,
    resume_path,
    save_resume_state,
)


def _state(epoch: int = 3) -> ResumeState:
    return ResumeState(
        epoch=epoch,
        model={"fc.weight": torch.zeros(2, 2)},
        optimizer={"state": {}, "param_groups": [{"lr": 0.001}]},
        scheduler={"last_epoch": epoch},
        scaler=None,
        best_metric=0.42,
        best_epoch=2,
        patience_counter=1,
        history=[{"epoch": 1}, {"epoch": 2}, {"epoch": 3}],
    )


class TestRoundTrip:
    def test_restores_every_field(self, tmp_path: Path) -> None:
        save_resume_state(tmp_path, _state())

        got = load_resume_state(tmp_path)

        assert got is not None
        assert got.epoch == 3
        assert got.best_metric == 0.42
        assert got.best_epoch == 2
        assert got.patience_counter == 1
        assert len(got.history) == 3
        assert got.scheduler == {"last_epoch": 3}
        assert torch.equal(got.model["fc.weight"], torch.zeros(2, 2))

    def test_writes_the_expected_filename(self, tmp_path: Path) -> None:
        save_resume_state(tmp_path, _state())

        assert (tmp_path / RESUME_FILENAME).is_file()

    def test_leaves_no_temporary_behind(self, tmp_path: Path) -> None:
        # The write is atomic; a stray .tmp would be found and read next time.
        save_resume_state(tmp_path, _state())

        assert list(tmp_path.glob("*.tmp")) == []

    def test_overwrites_cleanly(self, tmp_path: Path) -> None:
        save_resume_state(tmp_path, _state(epoch=3))
        save_resume_state(tmp_path, _state(epoch=7))

        got = load_resume_state(tmp_path)

        assert got is not None and got.epoch == 7


class TestRefusesRatherThanGuesses:
    def test_absent_state_is_none(self, tmp_path: Path) -> None:
        assert load_resume_state(tmp_path) is None

    def test_unreadable_file_is_none_rather_than_an_exception(
        self, tmp_path: Path
    ) -> None:
        # A run that cannot continue is smaller than one that refuses to start.
        resume_path(tmp_path).write_bytes(b"not a torch file")

        assert load_resume_state(tmp_path) is None

    def test_a_different_format_version_is_refused(self, tmp_path: Path) -> None:
        torch.save({"format_version": 99, "epoch": 3}, resume_path(tmp_path))

        assert load_resume_state(tmp_path) is None

    def test_missing_fields_are_refused(self, tmp_path: Path) -> None:
        torch.save({"format_version": 1, "epoch": 3}, resume_path(tmp_path))

        assert load_resume_state(tmp_path) is None

    def test_a_non_dict_payload_is_refused(self, tmp_path: Path) -> None:
        torch.save([1, 2, 3], resume_path(tmp_path))

        assert load_resume_state(tmp_path) is None


class TestClear:
    def test_removes_the_file(self, tmp_path: Path) -> None:
        save_resume_state(tmp_path, _state())

        clear_resume_state(tmp_path)

        assert load_resume_state(tmp_path) is None

    def test_is_harmless_when_there_is_nothing(self, tmp_path: Path) -> None:
        clear_resume_state(tmp_path)  # must not raise


class TestCanResume:
    def test_true_when_it_stopped_short(self, tmp_path: Path) -> None:
        save_resume_state(tmp_path, _state(epoch=3))

        assert can_resume(tmp_path, configured_epochs=10) is True

    def test_false_when_it_reached_the_end(self, tmp_path: Path) -> None:
        save_resume_state(tmp_path, _state(epoch=10))

        assert can_resume(tmp_path, configured_epochs=10) is False

    def test_false_without_state(self, tmp_path: Path) -> None:
        assert can_resume(tmp_path, configured_epochs=10) is False
