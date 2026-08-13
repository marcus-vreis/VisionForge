"""Tests for the FIFO run queue (ADR-075).

Driven with ``asyncio.run`` rather than an async-test plugin (the project has
none), which is enough: every behavior here is about ordering and bookkeeping
inside one event loop.
"""

from __future__ import annotations

import asyncio

from visionforge.gui.api.run_queue import QueuedJob, RunQueue


def _job(run_id: str, body, **extra) -> QueuedJob:
    return QueuedJob(
        run_id=run_id,
        label=extra.get("label", run_id),
        task=extra.get("task", "classification"),
        strategy=extra.get("strategy", "simple"),
        start=body,
    )


def _queue(on_finish=None) -> tuple[RunQueue, list[str]]:
    """A queue plus the ordered log of jobs it started."""
    started: list[str] = []
    queue = RunQueue(
        on_start=lambda job: started.append(job.run_id),
        on_finish=on_finish
        or (lambda job: {"run_id": job.run_id, "status": "completed"}),
    )
    return queue, started


class TestOrdering:
    def test_first_submission_runs_immediately(self) -> None:
        async def scenario() -> str:
            queue, _ = _queue()

            async def body() -> None:
                await asyncio.sleep(0)

            status = queue.submit(_job("a", body))
            await asyncio.sleep(0.05)
            return status

        assert asyncio.run(scenario()) == "running"

    def test_second_submission_is_queued_not_rejected(self) -> None:
        async def scenario() -> tuple[str, str, int]:
            queue, _ = _queue()
            gate = asyncio.Event()

            async def blocking() -> None:
                await gate.wait()

            async def quick() -> None:
                return None

            first = queue.submit(_job("a", blocking))
            await asyncio.sleep(0.01)  # let the worker pick "a" up
            second = queue.submit(_job("b", quick))
            depth = queue.pending_count()
            gate.set()
            await asyncio.sleep(0.05)
            return first, second, depth

        first, second, depth = asyncio.run(scenario())
        assert (first, second, depth) == ("running", "queued", 1)

    def test_jobs_execute_in_submission_order(self) -> None:
        async def scenario() -> list[str]:
            queue, started = _queue()
            gate = asyncio.Event()
            done: list[str] = []

            async def blocking() -> None:
                await gate.wait()
                done.append("a")

            def make(name: str):
                async def body() -> None:
                    done.append(name)

                return body

            queue.submit(_job("a", blocking))
            await asyncio.sleep(0.01)
            for name in ("b", "c", "d"):
                queue.submit(_job(name, make(name)))
            gate.set()
            await asyncio.sleep(0.1)
            assert started == ["a", "b", "c", "d"]
            return done

        assert asyncio.run(scenario()) == ["a", "b", "c", "d"]

    def test_only_one_job_runs_at_a_time(self) -> None:
        """The machine has one GPU: overlap would be the bug this prevents."""

        async def scenario() -> int:
            queue, _ = _queue()
            concurrent = 0
            peak = 0

            def make():
                async def body() -> None:
                    nonlocal concurrent, peak
                    concurrent += 1
                    peak = max(peak, concurrent)
                    await asyncio.sleep(0.02)
                    concurrent -= 1

                return body

            for name in ("a", "b", "c"):
                queue.submit(_job(name, make()))
            await asyncio.sleep(0.2)
            return peak

        assert asyncio.run(scenario()) == 1


class TestCancel:
    def test_pending_job_can_be_cancelled(self) -> None:
        async def scenario() -> tuple[bool, list[str]]:
            queue, started = _queue()
            gate = asyncio.Event()

            async def blocking() -> None:
                await gate.wait()

            async def never() -> None:  # pragma: no cover - must not run
                raise AssertionError("cancelled job executed")

            queue.submit(_job("a", blocking))
            await asyncio.sleep(0.01)
            queue.submit(_job("b", never))
            cancelled = queue.cancel("b")
            gate.set()
            await asyncio.sleep(0.05)
            return cancelled, started

        cancelled, started = asyncio.run(scenario())
        assert cancelled is True
        assert started == ["a"]

    def test_running_job_is_asked_to_stop(self) -> None:
        """ADR-088 gave the trainers a safe stop point: the epoch boundary."""

        async def scenario() -> bool:
            queue, _ = _queue()
            gate = asyncio.Event()

            async def blocking() -> None:
                await gate.wait()

            job = _job("a", blocking)
            queue.submit(job)
            await asyncio.sleep(0.01)
            result = queue.cancel("a")
            gate.set()
            await asyncio.sleep(0.05)
            # True means the request reached the run, not that it already ended.
            return result and job.cancel_token.cancelled

        assert asyncio.run(scenario()) is True

    def test_unknown_id_cancels_nothing(self) -> None:
        queue, _ = _queue()
        assert queue.cancel("nope") is False


class TestSnapshotAndRecords:
    def test_snapshot_shows_active_and_pending(self) -> None:
        async def scenario() -> dict:
            queue, _ = _queue()
            gate = asyncio.Event()

            async def blocking() -> None:
                await gate.wait()

            async def quick() -> None:
                return None

            queue.submit(_job("a", blocking, label="first", task="detection"))
            await asyncio.sleep(0.01)
            queue.submit(_job("b", quick, label="second", strategy="sweep"))
            snap = queue.snapshot()
            gate.set()
            await asyncio.sleep(0.05)
            return snap

        snap = asyncio.run(scenario())
        assert snap["active"]["run_id"] == "a"
        assert snap["active"]["task"] == "detection"
        assert [p["run_id"] for p in snap["pending"]] == ["b"]
        assert snap["pending"][0]["strategy"] == "sweep"
        assert "start" not in snap["active"]  # the callable never leaks into JSON

    def test_finished_run_survives_the_next_one_starting(self) -> None:
        """The reason a record store exists: /experiment/result must still answer."""

        async def scenario() -> tuple[dict | None, str | None]:
            reports = {"a": "report-a", "b": "report-b"}
            queue = RunQueue(
                on_start=lambda job: None,
                on_finish=lambda job: {
                    "run_id": job.run_id,
                    "status": "completed",
                    "report": reports[job.run_id],
                },
            )

            async def quick() -> None:
                return None

            queue.submit(_job("a", quick))
            queue.submit(_job("b", quick))
            await asyncio.sleep(0.1)
            record = queue.record("a")
            return record, queue.active_run_id

        record, active = asyncio.run(scenario())
        assert record is not None
        assert record["report"] == "report-a"
        assert active is None

    def test_unrecorded_run_reads_as_none(self) -> None:
        queue, _ = _queue()
        assert queue.record("never-submitted") is None

    def test_records_are_bounded(self) -> None:
        """A long session must not accumulate reports forever."""
        from visionforge.gui.api import run_queue as module

        async def scenario() -> int:
            queue, _ = _queue()

            async def quick() -> None:
                return None

            for i in range(module._MAX_RECORDS + 10):
                queue.submit(_job(f"r{i}", quick))
            await asyncio.sleep(0.5)
            return len(queue._records)

        assert asyncio.run(scenario()) == module._MAX_RECORDS


class TestFailureIsolation:
    def test_a_raising_job_does_not_strand_the_queue(self) -> None:
        """Submitting a batch is pointless if one bad job stops the rest."""

        async def scenario() -> list[str]:
            queue, started = _queue()
            ran: list[str] = []

            async def boom() -> None:
                raise RuntimeError("kaboom")

            async def fine() -> None:
                ran.append("b")

            queue.submit(_job("a", boom))
            queue.submit(_job("b", fine))
            await asyncio.sleep(0.1)
            assert started == ["a", "b"]
            return ran

        assert asyncio.run(scenario()) == ["b"]

    def test_a_finished_worker_is_replaced_on_the_next_submit(self) -> None:
        """The drain task exits when the queue empties; a later submit revives it."""

        async def scenario() -> list[str]:
            queue, started = _queue()

            async def quick() -> None:
                return None

            queue.submit(_job("a", quick))
            await asyncio.sleep(0.05)
            assert not queue.is_busy()
            queue.submit(_job("b", quick))
            await asyncio.sleep(0.05)
            return started

        assert asyncio.run(scenario()) == ["a", "b"]


# ── HTTP surface ──────────────────────────────────────────────────────────────


class TestQueueEndpoints:
    """GET /api/queue and DELETE /api/queue/{run_id}."""

    @staticmethod
    def _client_and_routes():
        from fastapi.testclient import TestClient

        import visionforge.gui.api.routes as routes_mod
        from visionforge.gui.server import app

        return TestClient(app, raise_server_exceptions=True), routes_mod

    def test_idle_queue_is_empty(self) -> None:
        from .conftest import release_queue

        client, routes_mod = self._client_and_routes()
        release_queue(routes_mod)

        body = client.get("/api/queue").json()

        assert body == {"active": None, "pending": []}

    def test_queue_lists_the_active_job_and_what_waits(self, tmp_path) -> None:
        from .conftest import occupy_queue, release_queue

        client, routes_mod = self._client_and_routes()
        occupy_queue(routes_mod, run_id="holding")
        try:
            submitted = client.post(
                "/api/regression/run", json=_regression_payload(tmp_path)
            )
            assert submitted.json()["status"] == "queued"

            body = client.get("/api/queue").json()

            assert body["active"]["run_id"] == "holding"
            assert len(body["pending"]) == 1
            assert body["pending"][0]["task"] == "regression"
            assert body["pending"][0]["strategy"] == "simple"
        finally:
            release_queue(routes_mod)

    def test_status_reports_how_many_are_waiting(self, tmp_path) -> None:
        from .conftest import occupy_queue, release_queue

        client, routes_mod = self._client_and_routes()
        occupy_queue(routes_mod)
        try:
            client.post("/api/regression/run", json=_regression_payload(tmp_path))
            client.post("/api/regression/run", json=_regression_payload(tmp_path))

            assert client.get("/api/experiment/status").json()["queued"] == 2
        finally:
            release_queue(routes_mod)

    def test_cancelling_a_pending_job_removes_it(self, tmp_path) -> None:
        from .conftest import occupy_queue, release_queue

        client, routes_mod = self._client_and_routes()
        occupy_queue(routes_mod)
        try:
            run_id = client.post(
                "/api/regression/run", json=_regression_payload(tmp_path)
            ).json()["run_id"]

            cancelled = client.delete(f"/api/queue/{run_id}")

            assert cancelled.status_code == 200
            assert cancelled.json()["status"] == "cancelled"
            assert client.get("/api/queue").json()["pending"] == []
        finally:
            release_queue(routes_mod)

    def test_cancelling_the_running_job_asks_it_to_stop(self) -> None:
        """ADR-088: a running job is now stoppable at its epoch boundary."""
        from .conftest import occupy_queue, release_queue

        client, routes_mod = self._client_and_routes()
        occupy_queue(routes_mod, run_id="holding")
        try:
            resp = client.delete("/api/queue/holding")

            assert resp.status_code == 200
            assert routes_mod._RUN_QUEUE._active.cancel_token.cancelled is True
        finally:
            release_queue(routes_mod)

    def test_the_running_jobs_token_is_published_for_its_executor(self) -> None:
        """Cancelling only works if the executor can find the token.

        The executor coroutine is built at submission time and cannot close over
        a job that does not exist yet, so it reads the active token from the
        module. Without this the DELETE flips a flag no trainer ever sees.
        """
        from .conftest import occupy_queue, release_queue

        client, routes_mod = self._client_and_routes()
        occupy_queue(routes_mod, run_id="holding")
        try:
            job = routes_mod._RUN_QUEUE._active
            routes_mod._begin_job(job)

            assert routes_mod._active_cancel_token is job.cancel_token

            client.delete("/api/queue/holding")
            assert routes_mod._active_cancel_token.cancelled is True
        finally:
            release_queue(routes_mod)

    def test_cancelling_an_unknown_id_is_a_404(self) -> None:
        from .conftest import release_queue

        client, routes_mod = self._client_and_routes()
        release_queue(routes_mod)

        assert client.delete("/api/queue/never-existed").status_code == 404


def _regression_payload(tmp_path) -> dict:
    base = tmp_path / "ds"
    base.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        (base / f"{split}.csv").write_text(
            "image,target\na.png,1.0\n", encoding="utf-8"
        )
    return {
        "name": "queued_reg",
        "model": {"name": "resnet18", "num_targets": 1, "pretrained": False},
        "data": {"base_dir": str(base), "num_workers": 0},
        "training": {"epochs": 1, "batch_size": 2, "learning_rate": 0.001},
        "output": {
            "models_dir": str(tmp_path / "models"),
            "reports_dir": str(tmp_path / "reports"),
            "graphics_dir": str(tmp_path / "graphics"),
            "logs_dir": str(tmp_path / "logs"),
        },
    }
