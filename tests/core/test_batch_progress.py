"""A long epoch reports how far into it the run is (the overlay used to freeze)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from visionforge.core import training_loop as loop
from visionforge.core.trainer import save_best_checkpoint


@dataclass
class ScoredEpoch:
    epoch: int
    score: float


class _SteppedClock:
    """Advances by a fixed step on every read, like one slow batch per call."""

    def __init__(self, step_s: float) -> None:
        self.now = 0.0
        self._step_s = step_s

    def __call__(self) -> float:
        self.now += self._step_s
        return self.now


def _loader(n_batches: int) -> DataLoader[Any]:
    data = TensorDataset(torch.arange(n_batches, dtype=torch.float32).unsqueeze(1))
    return DataLoader(data, batch_size=1, generator=torch.Generator().manual_seed(0))


def test_a_slow_epoch_reports_its_batches_and_leaves_the_last_to_epoch_end() -> None:
    events: list[dict[str, Any]] = []
    wrapped = loop.BatchProgressLoader(
        _loader(5),
        loop.BatchProgressReporter(
            events.append, 10, interval_s=2.0, clock=_SteppedClock(1.0)
        ),
    )
    wrapped.epoch = 3

    batches = [b[0].item() for b in wrapped]

    assert batches == [0.0, 1.0, 2.0, 3.0, 4.0]
    # One read per batch at 1 s each: 2 s have passed after batches 2 and 4,
    # and batch 5 is the epoch's last, which epoch_end covers.
    assert [e["batch"] for e in events] == [2, 4]
    assert events[0] == {
        "event": "batch",
        "epoch": 3,
        "batch": 2,
        "total_batches": 5,
        "total_epochs": 10,
    }


def test_a_fast_epoch_adds_nothing_to_the_stream() -> None:
    events: list[dict[str, Any]] = []
    wrapped = loop.BatchProgressLoader(
        _loader(5),
        loop.BatchProgressReporter(
            events.append, 10, interval_s=2.0, clock=_SteppedClock(0.01)
        ),
    )
    assert len(list(wrapped)) == 5
    assert events == []


def test_the_wrapper_still_looks_like_the_loader() -> None:
    inner = _loader(3)
    wrapped = loop.with_batch_progress(inner, lambda _e: None, 1)
    assert len(wrapped) == 3
    assert wrapped.num_workers == inner.num_workers
    # Resume reads the generators through these attributes.
    assert wrapped.generator is inner.generator


def test_a_loader_without_length_is_iterated_untouched() -> None:
    """Custom tasks (ADR-058) may hand the engine a generator."""
    events: list[dict[str, Any]] = []
    wrapped = loop.with_batch_progress(iter([1, 2, 3]), events.append, 2)
    assert list(wrapped) == [1, 2, 3]
    assert events == []


class _FakeDataModule:
    def train_loader(self) -> DataLoader[Any]:
        return _loader(4)

    def val_loader(self) -> DataLoader[Any]:
        return _loader(1)


def test_run_epochs_stamps_each_epoch_on_the_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resumed run starting at epoch 2 reports epochs 2 and 3, not 1 and 2."""
    monkeypatch.setattr(loop, "BATCH_PROGRESS_INTERVAL_S", 0.0)
    events: list[dict[str, Any]] = []
    train, val, workers = loop.start_training(
        _FakeDataModule(), events.append, 3, "cpu"
    )
    assert isinstance(train, loop.BatchProgressLoader)
    assert workers == {"train": 0, "val": 0}
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    def compute(epoch: int) -> ScoredEpoch:
        for _ in train:
            pass
        return ScoredEpoch(epoch, 1.0)

    writes, _stopped = loop.run_epochs(
        loop.EpochState[ScoredEpoch]([], float("inf"), 0, 0, 2),
        loop.EpochRuntime(train, val, 3, None, None),
        loop.CheckpointContext(
            model,
            tmp_path / "best_model.pth",
            save_best_checkpoint,
            lambda: loop.checkpoint_components(model, optimizer, None, None),
        ),
        loop.EpochCallbacks(compute, lambda _r: None, lambda _c: False),
        loop.MetricPolicy("score"),
    )
    if writes.future is not None:
        writes.future.result()
    writes.pool.shutdown()

    batch_epochs = [(e["epoch"], e["batch"]) for e in events if e["event"] == "batch"]
    assert batch_epochs == [(2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)]


def test_no_callback_means_the_plain_loader() -> None:
    train, _val, _workers = loop.start_training(_FakeDataModule(), None, 3, "cpu")
    assert isinstance(train, DataLoader)
