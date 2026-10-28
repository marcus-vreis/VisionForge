"""Shared epoch contexts preserve checkpoint state and execution boundaries."""

from __future__ import annotations

import random
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from visionforge.core import training_loop as loop
from visionforge.core.cancellation import CancellationToken
from visionforge.core.resume import ResumeState, load_resume_state
from visionforge.core.trainer import save_best_checkpoint


@dataclass
class ScoredEpoch:
    """Keep controlled ranking separate from the real optimizer trajectory."""

    epoch: int
    score: float


def _loader() -> DataLoader:
    return DataLoader(
        TensorDataset(torch.tensor([[1.0], [2.0]]), torch.zeros(2, 1)),
        batch_size=2,
        generator=torch.Generator().manual_seed(23),
    )


@pytest.mark.parametrize("higher_is_better", [False, True])
def test_epoch_contexts_preserve_completed_state_and_best_weights(
    tmp_path: Path, higher_is_better: bool
) -> None:
    """Ties and worsening scores retain the best epoch and advance patience."""
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    scaler = torch.amp.GradScaler("cpu")
    train_loader, val_loader = _loader(), _loader()
    runtime = loop.EpochRuntime(train_loader, val_loader, 8, None, None)
    state = loop.EpochState[ScoredEpoch](
        [], -float("inf") if higher_is_better else float("inf"), 0, 0, 1
    )
    scores = [0.1, 0.3, 0.3, 0.2] if higher_is_better else [3.0, 2.0, 2.0, 4.0]
    weights: dict[int, torch.Tensor] = {}
    reported: list[tuple[int, int, int]] = []

    def compute(epoch: int) -> ScoredEpoch:
        inputs, targets = next(iter(train_loader))
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(inputs), targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        weights[epoch] = model.weight.detach().clone()
        return ScoredEpoch(epoch, scores[epoch - 1])

    checkpoint = loop.CheckpointContext(
        model,
        tmp_path / "best_model.pth",
        save_best_checkpoint,
        lambda: loop.checkpoint_components(model, optimizer, scheduler, scaler),
    )
    writes, stopped = loop.run_epochs(
        state,
        runtime,
        checkpoint,
        loop.EpochCallbacks(
            compute,
            lambda result: reported.append(
                (result.epoch, len(state.history), state.best_epoch)
            ),
            lambda counter: counter >= 2,
        ),
        loop.MetricPolicy("score", higher_is_better),
    )
    try:
        assert writes.future is not None
        writes.future.result()
        saved = load_resume_state(tmp_path)
        assert saved is not None
        assert not stopped
        assert [r.epoch for r in state.history] == [1, 2, 3, 4]
        assert reported == [(1, 1, 0), (2, 2, 1), (3, 3, 2), (4, 4, 2)]
        assert saved.epoch == 4
        assert saved.best_epoch == state.best_epoch == 2
        assert saved.best_metric == state.best_metric == scores[1]
        assert saved.patience_counter == state.patience_counter == 2
        assert saved.history == [asdict(r) for r in state.history]
        assert saved.scheduler == scheduler.state_dict()
        assert saved.scaler == scaler.state_dict()
        assert saved.optimizer["param_groups"] == optimizer.state_dict()["param_groups"]
        assert torch.equal(
            saved.optimizer["state"][0]["momentum_buffer"],
            optimizer.state_dict()["state"][0]["momentum_buffer"],
        )
        assert torch.equal(saved.model["weight"], weights[4])
        best = torch.load(checkpoint.model_path, weights_only=True)
        assert torch.equal(best["weight"], weights[2])
        assert not torch.equal(best["weight"], weights[4])
        assert saved.rng_state is not None
    finally:
        writes.pool.shutdown(wait=True)


@pytest.mark.parametrize("boundary", ["patience", "cancelled", "exhausted"])
def test_epoch_contexts_skip_computation_at_completed_boundaries(
    tmp_path: Path, boundary: str
) -> None:
    """A restored stopping boundary or cancellation never trains another epoch."""
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    token = CancellationToken()
    if boundary == "cancelled":
        token.cancel()
    history = [ScoredEpoch(1, 0.2)]
    state = loop.EpochState(history, 0.2, 1, 2 if boundary == "patience" else 0, 2)

    def unexpected_compute(epoch: int) -> ScoredEpoch:
        raise AssertionError(f"Unexpected epoch {epoch} after {boundary}")

    def unexpected_report(result: ScoredEpoch) -> None:
        pytest.fail(f"Unexpected report {result} after {boundary}")

    writes, stopped = loop.run_epochs(
        state,
        loop.EpochRuntime(
            _loader(), _loader(), 1 if boundary == "exhausted" else 3, None, token
        ),
        loop.CheckpointContext(
            model,
            tmp_path / "best_model.pth",
            save_best_checkpoint,
            lambda: loop.checkpoint_components(model, optimizer, None),
        ),
        loop.EpochCallbacks(unexpected_compute, unexpected_report, lambda n: n >= 2),
        loop.MetricPolicy("score"),
    )
    try:
        assert stopped is (boundary == "cancelled")
        assert writes.future is None
        assert state.history == [ScoredEpoch(1, 0.2)]
        assert state.best_epoch == 1
        assert state.best_metric == 0.2
        assert not list(tmp_path.iterdir())
    finally:
        writes.pool.shutdown(wait=True)


def test_checkpoint_writer_failure_is_propagated(tmp_path: Path) -> None:
    """An asynchronous disk failure reaches the caller at checkpoint completion."""
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    missing_parent = tmp_path / "absent" / "best_model.pth"

    def write_to_missing_parent(
        pool: ThreadPoolExecutor,
        pending: Future[None] | None,
        model: nn.Module,
        path: Path,
    ) -> Future[None]:
        return save_best_checkpoint(pool, pending, model, missing_parent)

    state = loop.EpochState[ScoredEpoch]([], float("inf"), 0, 0, 1)
    writes, _ = loop.run_epochs(
        state,
        loop.EpochRuntime(_loader(), _loader(), 1, None, None),
        loop.CheckpointContext(
            model,
            tmp_path / "best_model.pth",
            write_to_missing_parent,
            lambda: loop.checkpoint_components(model, optimizer, None),
        ),
        loop.EpochCallbacks(
            lambda e: ScoredEpoch(e, 0.1), lambda _: None, lambda _: False
        ),
        loop.MetricPolicy("score"),
    )
    try:
        with pytest.raises(RuntimeError, match="does not exist"):
            loop.finish_checkpoint_writes(tmp_path, None, writes.pool, writes.future)
        assert not (tmp_path / "best_model.pth").exists()
    finally:
        writes.pool.shutdown(wait=True)


def _assert_same_state(actual: Any, expected: Any) -> None:
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_same_state(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for value, reference in zip(actual, expected, strict=True):
            _assert_same_state(value, reference)
    else:
        assert actual == expected


@pytest.mark.parametrize("higher_is_better", [False, True])
def test_resumed_contexts_preserve_full_checkpoint_trajectory(
    tmp_path: Path, higher_is_better: bool
) -> None:
    """Resume restores SGD momentum, scheduler, scaler and every CPU RNG exactly."""

    def train(
        path: Path, *, resumed: ResumeState | None = None, interrupt: bool = False
    ) -> ResumeState:
        path.mkdir(exist_ok=True)
        random.seed(17)
        np.random.seed(17)
        torch.manual_seed(17)
        model = nn.Sequential(nn.Dropout(0.3), nn.Linear(1, 1))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        scaler = torch.amp.GradScaler("cpu")
        restored = loop.restore_training_state(
            resumed,
            model,
            optimizer,
            scheduler,
            path,
            4,
            ScoredEpoch,
            -float("inf") if higher_is_better else float("inf"),
            scaler,
        )
        state = loop.EpochState(*restored)
        token = CancellationToken()
        train_loader, val_loader = _loader(), _loader()

        def compute(epoch: int) -> ScoredEpoch:
            inputs, targets = next(iter(train_loader))
            noise = random.random() + float(np.random.random()) + torch.rand(())
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(inputs + noise), targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if interrupt and epoch == 2:
                token.cancel()
            return ScoredEpoch(epoch, float(loss.detach()))

        writes, stopped = loop.run_epochs(
            state,
            loop.EpochRuntime(train_loader, val_loader, 4, resumed, token),
            loop.CheckpointContext(
                model,
                path / "best_model.pth",
                save_best_checkpoint,
                lambda: loop.checkpoint_components(model, optimizer, scheduler, scaler),
            ),
            loop.EpochCallbacks(compute, lambda _: None, lambda _: False),
            loop.MetricPolicy("score", higher_is_better),
        )
        try:
            assert stopped is interrupt
            if writes.future is not None:
                writes.future.result()
            saved = load_resume_state(path)
            assert saved is not None
            return saved
        finally:
            writes.pool.shutdown(wait=True)

    continuous = train(tmp_path / "continuous")
    interrupted = train(tmp_path / "resumed", interrupt=True)
    assert interrupted.epoch == 2
    assert interrupted.optimizer["state"][0]["momentum_buffer"].numel() > 0
    assert interrupted.scheduler is not None
    assert interrupted.scheduler["last_epoch"] == 2
    assert interrupted.scaler is not None
    assert interrupted.scaler["_growth_tracker"] == 2
    assert interrupted.rng_state is not None
    assert interrupted.rng_state["loaders"]
    expected_cuda = torch.cuda.device_count() if torch.cuda.is_initialized() else 0
    assert len(interrupted.rng_state["cuda"]) == expected_cuda
    resumed = train(tmp_path / "resumed", resumed=interrupted)
    assert resumed.epoch == continuous.epoch == 4
    _assert_same_state(asdict(resumed), asdict(continuous))
    _assert_same_state(
        torch.load(tmp_path / "resumed" / "best_model.pth", weights_only=True),
        torch.load(tmp_path / "continuous" / "best_model.pth", weights_only=True),
    )
