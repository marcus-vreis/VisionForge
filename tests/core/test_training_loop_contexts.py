"""Shared epoch contexts preserve checkpoint state and execution boundaries."""

from __future__ import annotations

import json
import random
import threading
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
        written: Future[None] = save_best_checkpoint(
            pool, pending, model, missing_parent
        )
        return written

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


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_metric_stops_instead_of_counting_as_no_improvement(
    tmp_path: Path, bad: float
) -> None:
    """A NaN/inf score used to add patience and save nothing; now it raises."""
    model = nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    runtime = loop.EpochRuntime(_loader(), _loader(), 8, None, None)
    state = loop.EpochState[ScoredEpoch]([], float("inf"), 0, 0, 1)
    checkpoint = loop.CheckpointContext(
        model,
        tmp_path / "best_model.pth",
        save_best_checkpoint,
        lambda: loop.checkpoint_components(model, optimizer, None),
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        writes = loop.CheckpointWrites(pool)
        with pytest.raises(loop.NonFiniteMetricError, match="non-finite") as caught:
            loop.save_epoch_checkpoint(
                bad, 3, state, checkpoint, writes, runtime, loop.MetricPolicy("score")
            )
    assert caught.value.epoch == 3
    assert caught.value.name == "score"
    assert str(bad) in str(caught.value)
    assert state.patience_counter == 0
    assert writes.future is None
    assert not checkpoint.model_path.exists()


def test_run_epochs_waits_for_pending_write_before_raising_non_finite(
    tmp_path: Path,
) -> None:
    """The epoch that improved is written to disk before the later NaN aborts."""
    model = nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    runtime = loop.EpochRuntime(_loader(), _loader(), 5, None, None)
    state = loop.EpochState[ScoredEpoch]([], float("inf"), 0, 0, 1)
    release = threading.Event()
    threading.Timer(0.3, release.set).start()
    finished: list[bool] = []

    def slow_save(
        pool: ThreadPoolExecutor,
        pending: Future[None] | None,
        net: nn.Module,
        path: Path,
    ) -> Future[None]:
        def write() -> None:
            release.wait(5)
            torch.save(net.state_dict(), path)
            finished.append(True)

        return pool.submit(write)

    checkpoint = loop.CheckpointContext(
        model,
        tmp_path / "best_model.pth",
        slow_save,
        lambda: loop.checkpoint_components(model, optimizer, None),
    )
    scores = {1: 2.0, 2: float("nan")}
    callbacks = loop.EpochCallbacks[ScoredEpoch](
        compute=lambda epoch: ScoredEpoch(epoch, scores[epoch]),
        report=lambda result: None,
        early_stop=lambda counter: False,
    )
    with pytest.raises(loop.NonFiniteMetricError) as caught:
        loop.run_epochs(
            state, runtime, checkpoint, callbacks, loop.MetricPolicy("score")
        )
    assert caught.value.epoch == 2
    assert [h.epoch for h in state.history] == [1, 2]
    # The write was still blocked when epoch 2 raised: only a wait on its
    # future can make it done by the time the exception arrives.
    assert finished == [True]
    assert checkpoint.model_path.is_file()
    assert caught.value.best_epoch == 1
    assert "checkpoint from epoch 1 was kept" in str(caught.value)


def test_non_finite_error_says_whether_a_checkpoint_exists() -> None:
    first = loop.NonFiniteMetricError("val_loss", 1, float("nan"))
    later = loop.NonFiniteMetricError("val_loss", 4, float("nan"), best_epoch=3)
    assert "no checkpoint was saved" in str(first)
    assert "was kept" not in str(first)
    assert "checkpoint from epoch 3 was kept" in str(later)
    assert "no checkpoint" not in str(later)
    assert later.health_warning().code == "non_finite_metric"
    assert "3" in later.health_warning().message
    assert "sem salvar" in first.health_warning().message or (
        "without saving" in first.health_warning().message
    )
    assert "sem salvar" not in later.health_warning().message
    assert "without saving" not in later.health_warning().message


def test_record_non_finite_failure_marks_run_json_failed(tmp_path: Path) -> None:
    """run.json keeps the history, says it failed and stays valid JSON."""
    (tmp_path / "run.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "metrics": {"best_val_loss": float("inf")},
                "history": [{"epoch": 1, "val_loss": float("nan")}],
            }
        ),
        encoding="utf-8",
    )
    error = loop.NonFiniteMetricError("val_loss", 1, float("nan"))
    loop.record_non_finite_failure(tmp_path, error)
    text = (tmp_path / "run.json").read_text(encoding="utf-8")
    data = json.loads(text, parse_constant=lambda c: pytest.fail(c))
    assert data["status"] == "failed"
    assert "non-finite" in data["error"]
    assert data["metrics"]["best_val_loss"] is None
    assert data["history"][0]["val_loss"] is None


def test_non_finite_metric_health_warning_names_epoch_and_metric() -> None:
    from visionforge.core.training_health import non_finite_metric

    warning = non_finite_metric("val_loss", 4, float("nan"))
    assert warning.code == "non_finite_metric"
    assert "val_loss" in warning.message
    assert "4" in warning.message


def test_run_epochs_rejects_nan_loss_even_when_the_ranking_metric_is_finite(
    tmp_path: Path,
) -> None:
    """mIoU of NaN logits is a number; the NaN loss still has to stop the run."""

    @dataclass
    class LossAndScore:
        epoch: int
        val_loss: float
        score: float

    model = nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    state = loop.EpochState[LossAndScore]([], -1.0, 0, 0, 1)
    checkpoint = loop.CheckpointContext(
        model,
        tmp_path / "best_model.pth",
        save_best_checkpoint,
        lambda: loop.checkpoint_components(model, optimizer, None),
    )
    callbacks = loop.EpochCallbacks[LossAndScore](
        compute=lambda epoch: LossAndScore(epoch, float("nan"), 0.1),
        report=lambda result: None,
        early_stop=lambda counter: False,
    )
    runtime = loop.EpochRuntime(_loader(), _loader(), 3, None, None)
    policy = loop.MetricPolicy("score", higher_is_better=True)
    with pytest.raises(loop.NonFiniteMetricError, match="val_loss=nan"):
        loop.run_epochs(state, runtime, checkpoint, callbacks, policy)
    assert not checkpoint.model_path.exists()
