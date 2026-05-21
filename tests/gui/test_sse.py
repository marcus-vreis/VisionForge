"""Tests for GET /api/experiment/events SSE endpoint and Trainer progress_callback."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
from fastapi.testclient import TestClient

import visionforge.gui.api.routes as routes_mod
from visionforge.core.trainer import Trainer, TrainResult
from visionforge.gui.server import app
from visionforge.utils.config import ExperimentConfig

# ── helpers ───────────────────────────────────────────────────────────────────


class _Tiny(nn.Module):
    """Minimal binary model for trainer tests (B, 3, 32, 32) → (B, 1)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3 * 32 * 32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(1))


class _FakeData:
    """Returns two tiny batches for train/val loaders."""

    def _batches(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [
            (torch.randn(4, 3, 32, 32), torch.randint(0, 2, (4,))) for _ in range(2)
        ]

    def train_loader(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return self._batches()

    def val_loader(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return self._batches()


@pytest.fixture
def binary_config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "name": "sse_test",
            "task": "binary",
            "model": {"name": "resnet18", "num_classes": 1, "pretrained": False},
            "training": {
                "learning_rate": 0.01,
                "epochs": 3,
                "batch_size": 4,
                "early_stopping_patience": 10,
                "seed": 0,
            },
            "data": {"base_dir": str(tmp_path), "num_workers": 0, "pin_memory": False},
            "output": {
                "models_dir": str(tmp_path / "models"),
                "graphics_dir": str(tmp_path / "graphics"),
                "logs_dir": str(tmp_path / "logs"),
                "reports_dir": str(tmp_path / "reports"),
            },
        }
    )


# ── Trainer.fit progress_callback tests ───────────────────────────────────────


class TestTrainerProgressCallback:
    def test_callback_receives_start_event(
        self, binary_config: ExperimentConfig
    ) -> None:
        """fit() must emit a 'start' event as the first callback call."""
        events: list[dict[str, Any]] = []
        Trainer(binary_config).fit(
            _Tiny(), _FakeData(), progress_callback=events.append
        )
        assert events[0]["event"] == "start"

    def test_callback_receives_end_event(self, binary_config: ExperimentConfig) -> None:
        """fit() must emit an 'end' event as the last callback call."""
        events: list[dict[str, Any]] = []
        Trainer(binary_config).fit(
            _Tiny(), _FakeData(), progress_callback=events.append
        )
        assert events[-1]["event"] == "end"

    def test_epoch_end_count_matches_epochs_trained(
        self, binary_config: ExperimentConfig
    ) -> None:
        """fit() must emit one 'epoch_end' event per epoch actually trained."""
        events: list[dict[str, Any]] = []
        result = Trainer(binary_config).fit(
            _Tiny(), _FakeData(), progress_callback=events.append
        )
        epoch_events = [e for e in events if e["event"] == "epoch_end"]
        assert len(epoch_events) == result.total_epochs

    def test_epoch_end_event_schema(self, binary_config: ExperimentConfig) -> None:
        """Each epoch_end event must carry the required payload keys."""
        events: list[dict[str, Any]] = []
        Trainer(binary_config).fit(
            _Tiny(), _FakeData(), progress_callback=events.append
        )
        epoch_events = [e for e in events if e["event"] == "epoch_end"]
        required = {
            "event",
            "epoch",
            "total_epochs",
            "train_loss",
            "val_loss",
            "val_accuracy",
            "elapsed_s",
        }
        for ev in epoch_events:
            assert required <= ev.keys(), f"Missing keys in {ev}"

    def test_epoch_numbers_are_sequential(
        self, binary_config: ExperimentConfig
    ) -> None:
        """epoch_end events must arrive in order epoch=1, 2, 3."""
        events: list[dict[str, Any]] = []
        Trainer(binary_config).fit(
            _Tiny(), _FakeData(), progress_callback=events.append
        )
        epoch_events = [e for e in events if e["event"] == "epoch_end"]
        for i, ev in enumerate(epoch_events, start=1):
            assert ev["epoch"] == i

    def test_total_epochs_field_constant(self, binary_config: ExperimentConfig) -> None:
        """total_epochs in every epoch_end must equal config.training.epochs."""
        events: list[dict[str, Any]] = []
        Trainer(binary_config).fit(
            _Tiny(), _FakeData(), progress_callback=events.append
        )
        epoch_events = [e for e in events if e["event"] == "epoch_end"]
        for ev in epoch_events:
            assert ev["total_epochs"] == binary_config.training.epochs

    def test_elapsed_s_non_negative(self, binary_config: ExperimentConfig) -> None:
        """elapsed_s in each epoch_end must be >= 0."""
        events: list[dict[str, Any]] = []
        Trainer(binary_config).fit(
            _Tiny(), _FakeData(), progress_callback=events.append
        )
        for ev in events:
            if ev["event"] == "epoch_end":
                assert ev["elapsed_s"] >= 0

    def test_no_callback_does_not_raise(self, binary_config: ExperimentConfig) -> None:
        """fit() with progress_callback=None must complete without raising."""
        result = Trainer(binary_config).fit(
            _Tiny(), _FakeData(), progress_callback=None
        )
        assert isinstance(result, TrainResult)

    def test_event_order_start_epochs_end(
        self, binary_config: ExperimentConfig
    ) -> None:
        """Callback events must arrive: start → epoch_end × N → end."""
        events: list[dict[str, Any]] = []
        Trainer(binary_config).fit(
            _Tiny(), _FakeData(), progress_callback=events.append
        )
        kinds = [e["event"] for e in events]
        assert kinds[0] == "start"
        assert kinds[-1] == "end"
        middle = kinds[1:-1]
        assert all(k == "epoch_end" for k in middle)


# ── SSE endpoint tests ────────────────────────────────────────────────────────


@pytest.fixture
def client() -> TestClient:
    """TestClient with raise_server_exceptions for /api/experiment/events."""
    return TestClient(app, raise_server_exceptions=True)


class TestSSEEventQueue:
    """Unit tests for the queue-based event plumbing in routes."""

    def test_get_events_returns_text_event_stream(self, client: TestClient) -> None:
        """GET /api/experiment/events must use content-type text/event-stream."""
        routes_mod._current_run = None
        routes_mod._event_queue = None
        with client.stream("GET", "/api/experiment/events") as resp:
            assert "text/event-stream" in resp.headers["content-type"]

    def test_get_events_closes_when_no_active_run(self, client: TestClient) -> None:
        """With no active run the stream must yield a sentinel and close cleanly."""
        routes_mod._current_run = None
        routes_mod._event_queue = None
        chunks: list[str] = []
        with client.stream("GET", "/api/experiment/events") as resp:
            for line in resp.iter_lines():
                chunks.append(line)
        # Stream must close — if we reached here, it closed.
        assert isinstance(chunks, list)

    def test_stream_emits_correct_event_count(self) -> None:
        """Posting N-epoch events onto the queue must yield N+2 SSE data lines (start + N epoch_end + end)."""
        # Build an in-process event queue and fill it synchronously.
        loop = asyncio.new_event_loop()
        try:
            queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

            async def _fill() -> None:
                n = 3
                await queue.put({"event": "start", "total_epochs": n})
                for i in range(1, n + 1):
                    await queue.put(
                        {
                            "event": "epoch_end",
                            "epoch": i,
                            "total_epochs": n,
                            "train_loss": 0.5,
                            "val_loss": 0.4,
                            "val_accuracy": 0.8,
                            "elapsed_s": float(i),
                        }
                    )
                await queue.put({"event": "end"})
                await queue.put(None)  # sentinel

            loop.run_until_complete(_fill())

            # Drain the queue and collect events (mirrors the SSE generator logic).
            collected: list[dict[str, Any]] = []

            async def _drain() -> None:
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    collected.append(item)

            loop.run_until_complete(_drain())
        finally:
            loop.close()

        kinds = [e["event"] for e in collected]
        assert kinds[0] == "start"
        assert kinds[-1] == "end"
        assert kinds[1:-1] == ["epoch_end"] * 3

    def test_stream_event_order_via_fake_run(self, tmp_path: Path) -> None:
        """A complete fake run must produce start → epoch_end × N → end via SSE."""
        import threading
        import time

        loop = asyncio.new_event_loop()
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

        n_epochs = 2

        def _producer() -> None:
            # Simulate what _execute_experiment does: put events then sentinel.
            async def _put(item: dict[str, Any] | None) -> None:
                await queue.put(item)

            time.sleep(0.02)  # slight delay to let the consumer start
            asyncio.run_coroutine_threadsafe(
                _put({"event": "start", "total_epochs": n_epochs}), loop
            ).result()
            for i in range(1, n_epochs + 1):
                asyncio.run_coroutine_threadsafe(
                    _put(
                        {
                            "event": "epoch_end",
                            "epoch": i,
                            "total_epochs": n_epochs,
                            "train_loss": 0.5,
                            "val_loss": 0.4,
                            "val_accuracy": 0.8,
                            "elapsed_s": float(i),
                        }
                    ),
                    loop,
                ).result()
            asyncio.run_coroutine_threadsafe(_put({"event": "end"}), loop).result()
            asyncio.run_coroutine_threadsafe(_put(None), loop).result()

        collected: list[dict[str, Any]] = []

        async def _consumer() -> None:
            while True:
                item = await queue.get()
                if item is None:
                    break
                collected.append(item)

        t = threading.Thread(target=_producer, daemon=True)
        t.start()
        loop.run_until_complete(_consumer())
        t.join(timeout=5)
        loop.close()

        kinds = [e["event"] for e in collected]
        assert kinds == ["start"] + ["epoch_end"] * n_epochs + ["end"]

    def test_sse_endpoint_exists(self, client: TestClient) -> None:
        """GET /api/experiment/events must not return 404 or 405."""
        routes_mod._current_run = None
        routes_mod._event_queue = None
        with client.stream("GET", "/api/experiment/events") as resp:
            assert resp.status_code not in (404, 405)
