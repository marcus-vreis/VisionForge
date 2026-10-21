"""Half a restored training state is worse than none: it looks continued."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from loguru import logger
from torch.utils.data import DataLoader

from visionforge.core.resume import (
    RESUME_FILENAME,
    ResumeState,
    can_resume,
    capture_rng_state,
    clear_resume_state,
    load_resume_state,
    restore_rng_state,
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


def test_rng_roundtrip_is_safe_and_restores_all_cpu_sources(tmp_path: Path) -> None:
    """The file can be safely loaded and restores streams, including cached Gaussians."""
    random.gauss(0, 1)
    np.random.normal()
    generator = torch.Generator().manual_seed(53)
    loader = DataLoader(list(range(8)), shuffle=True, generator=generator)
    state = _state()
    state.rng_state = capture_rng_state(loader)
    save_resume_state(tmp_path, state)
    expected = (random.gauss(0, 1), np.random.normal(), torch.rand(4), list(loader))
    assert torch.load(resume_path(tmp_path), weights_only=True)["format_version"] == 2
    loaded = load_resume_state(tmp_path)
    assert loaded is not None
    restore_rng_state(loaded.rng_state, loader)
    assert random.gauss(0, 1) == expected[0]
    assert np.random.normal() == expected[1]
    assert torch.equal(torch.rand(4), expected[2])
    assert list(loader) == expected[3]


def test_legacy_resume_retains_training_state_and_warns(tmp_path: Path) -> None:
    """Version 1 remains resumable without claiming a reproducible trajectory."""
    save_resume_state(tmp_path, _state())
    payload = torch.load(resume_path(tmp_path), weights_only=True)
    payload["format_version"] = 1
    del payload["rng_state"]
    torch.save(payload, resume_path(tmp_path))
    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(str(message)))
    try:
        state = load_resume_state(tmp_path)
    finally:
        logger.remove(sink)
    assert state is not None and state.epoch == 3 and state.rng_state is None
    assert any("Legacy resume" in message for message in messages)
    restore_rng_state(None)


@pytest.mark.parametrize("field", ["python", "numpy", "torch", "cuda", "loaders"])
def test_corrupt_rng_is_refused_without_mutating_process(
    tmp_path: Path, field: str
) -> None:
    """A malformed RNG snapshot cannot partially reset the process."""
    save_resume_state(tmp_path, _state())
    payload = torch.load(resume_path(tmp_path), weights_only=True)
    payload["rng_state"][field] = "corrupt"
    torch.save(payload, resume_path(tmp_path))
    before = capture_rng_state()
    assert load_resume_state(tmp_path) is None
    after = capture_rng_state()
    assert before["python"] == after["python"]
    assert before["numpy"] == after["numpy"]
    assert torch.equal(before["torch"], after["torch"])


@pytest.mark.parametrize("failure", ["write", "replace"])
def test_atomic_save_failure_preserves_previous_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """An interrupted write or replacement cannot damage a valid checkpoint."""
    save_resume_state(tmp_path, _state(3))
    before = resume_path(tmp_path).read_bytes()

    def fail_write(payload: Any, path: Path) -> None:
        path.write_bytes(b"partial write")
        raise OSError("disk full")

    def fail_replace(source: Path, target: Path) -> None:
        raise OSError("replace denied")

    if failure == "write":
        monkeypatch.setattr("visionforge.core.resume.torch.save", fail_write)
    else:
        monkeypatch.setattr("visionforge.core.resume.os.replace", fail_replace)
    with pytest.raises(OSError):
        save_resume_state(tmp_path, _state(4))
    assert resume_path(tmp_path).read_bytes() == before
    loaded = load_resume_state(tmp_path)
    assert loaded is not None and loaded.epoch == 3
    assert not list(tmp_path.glob("*.tmp"))


def test_missing_loader_generator_is_rejected_before_rng_restore() -> None:
    """Changing loader generator topology cannot silently claim exact resume."""
    loader = DataLoader([1], generator=torch.Generator().manual_seed(1))
    state = capture_rng_state(loader)
    before = torch.get_rng_state()
    with pytest.raises(ValueError, match="loader generators differ"):
        restore_rng_state(state)
    assert torch.equal(torch.get_rng_state(), before)


def test_capture_does_not_initialize_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """A CPU checkpoint must not initialize a CUDA context to obtain RNG state."""
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)

    def unexpected_cuda_access() -> None:
        pytest.fail("CUDA RNG accessed before initialization")

    monkeypatch.setattr(torch.cuda, "get_rng_state_all", unexpected_cuda_access)
    assert capture_rng_state()["cuda"] == []


def test_initialized_cuda_rng_is_captured_and_restored_without_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise CUDA routing using synthetic byte states, without allocating on GPU."""
    cuda_state = torch.tensor([1, 2, 3], dtype=torch.uint8)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [cuda_state])
    state = capture_rng_state()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    restored: list[torch.Tensor] = []
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", restored.extend)
    restore_rng_state(state)
    assert len(restored) == 1 and torch.equal(restored[0], cuda_state)


def test_incompatible_cuda_and_worker_rng_limits_are_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A different device layout or worker processes cannot promise exact resume."""
    loader = DataLoader([1], num_workers=1)
    state = capture_rng_state(loader)
    state["cuda"] = [torch.tensor([1], dtype=torch.uint8)]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(str(message)))
    try:
        restore_rng_state(state, loader)
    finally:
        logger.remove(sink)
    assert any("CUDA RNG state cannot be restored" in message for message in messages)
    assert any("num_workers=0" in message for message in messages)
