from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from visionforge.core.tracking import TensorBoardLogger


def _install_fake_summary_writer(
    monkeypatch: pytest.MonkeyPatch, calls: list[tuple[str, float, int]]
) -> None:
    """Inject a fake torch.utils.tensorboard.SummaryWriter so no extra is needed."""

    class FakeWriter:
        def __init__(self, log_dir: str) -> None:
            self.log_dir = log_dir

        def add_scalar(self, tag: str, value: float, step: int) -> None:
            calls.append((tag, value, step))

        def flush(self) -> None:
            pass

        def close(self) -> None:
            pass

    fake_mod = types.ModuleType("torch.utils.tensorboard")
    fake_mod.SummaryWriter = FakeWriter  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", fake_mod)


class TestTensorBoardLogger:
    def test_writes_scalars_and_skips_none(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        calls: list[tuple[str, float, int]] = []
        _install_fake_summary_writer(monkeypatch, calls)

        tb = TensorBoardLogger(tmp_path / "tb")
        assert tb.enabled
        tb.log_scalars(1, {"loss/train": 0.5, "loss/val": 0.7, "skip": None})
        tb.close()

        assert ("loss/train", 0.5, 1) in calls
        assert ("loss/val", 0.7, 1) in calls
        assert all(tag != "skip" for tag, _, _ in calls)

    def test_noop_when_tensorboard_unavailable(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # A module without SummaryWriter makes the import raise -> disabled, no error.
        fake_mod = types.ModuleType("torch.utils.tensorboard")
        monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", fake_mod)

        tb = TensorBoardLogger(tmp_path / "tb")
        assert not tb.enabled
        tb.log_scalars(1, {"loss/train": 0.5})  # must not raise
        tb.close()
