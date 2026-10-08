"""The queue says where each job stops, and refuses a stop it cannot deliver (ADR-111).

ADR-094 was a DELETE that answered 200 while the trainer never saw the token.
The same thing was still true for K-fold, comparison, replicates, sweeps of the
standalone tasks and custom tasks. These tests pin the two halves of the fix at
the route layer: every executor hands the job's token to what it drives, and
``GET /api/queue`` names the boundary — or answers ``null`` and DELETE says 409.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
import torch
from fastapi.testclient import TestClient
from torch import nn

from visionforge.core.cancellation import STOPPED, CancellationToken
from visionforge.core.comparison import ComparisonTrial
from visionforge.core.replicates import ReplicateTrial
from visionforge.core.sweep import SweepTrial
from visionforge.core.task_runner import RunResult
from visionforge.gui.api.run_queue import NotStoppableError, QueuedJob, RunQueue
from visionforge.tasks import (
    BaseTaskConfig,
    TaskSpec,
    clear_task_registry,
    register_task,
)
from visionforge.utils.selftest_data import (
    build_anomaly_dataset,
    build_classification_dataset,
    build_regression_dataset,
)

from .conftest import occupy_queue, release_queue


@pytest.fixture
def client_and_routes():  # type: ignore[no-untyped-def]
    import visionforge.gui.api.routes as routes_mod
    from visionforge.gui.server import app

    occupy_queue(routes_mod, run_id="holding")
    try:
        yield TestClient(app, raise_server_exceptions=True), routes_mod
    finally:
        release_queue(routes_mod)
        routes_mod._active_cancel_token = None


@pytest.fixture(autouse=True)
def _clean_registry():  # type: ignore[no-untyped-def]
    clear_task_registry()
    yield
    clear_task_registry()


def _output(tmp_path: Path) -> dict[str, str]:
    return {
        "models_dir": str(tmp_path / "models"),
        "reports_dir": str(tmp_path / "reports"),
        "graphics_dir": str(tmp_path / "graphics"),
        "logs_dir": str(tmp_path / "logs"),
    }


def _classification(tmp_path: Path, **extra: Any) -> dict[str, Any]:
    base = build_classification_dataset(tmp_path / "cls")
    return {
        "name": "q_cls",
        "model": {"name": "resnet18", "num_classes": 2, "pretrained": False},
        "data": {"base_dir": str(base), "num_workers": 0},
        "output": _output(tmp_path),
        **extra,
    }


def _regression(tmp_path: Path) -> dict[str, Any]:
    base = build_regression_dataset(tmp_path / "reg")
    return {
        "name": "q_reg",
        "model": {"name": "resnet18", "num_targets": 1, "pretrained": False},
        "data": {"base_dir": str(base), "target_columns": ["target"]},
        "output": _output(tmp_path),
    }


def _anomaly(tmp_path: Path, model: str) -> dict[str, Any]:
    base = build_anomaly_dataset(tmp_path / "anom")
    return {
        "name": f"q_{model}",
        "model": {"name": model},
        "data": {"base_dir": str(base)},
        "output": _output(tmp_path),
    }


def _register_custom(key: str, *, owns_loop: bool) -> None:
    class _Spec(TaskSpec):
        def build_model(self, cfg: Any) -> nn.Module:
            return nn.Linear(3, 1)

        def build_loaders(self, cfg: Any) -> Any:
            batches = [(torch.randn(4, 3), torch.randn(4, 1))]
            return batches, batches, None

        def compute_loss(self, model: nn.Module, batch: Any, cfg: Any) -> Any:
            inputs, targets = batch
            return nn.functional.mse_loss(model(inputs), targets)

        def compute_metrics(self, model: nn.Module, loader: Any, cfg: Any) -> Any:
            return {"mae": 1.0}

    if owns_loop:

        class _Level2(_Spec):
            def run(self, cfg: Any, ctx: Any) -> dict[str, float] | None:
                return {"mae": 1.0}

        spec: type[TaskSpec] = _Level2
    else:
        spec = _Spec
    register_task(
        key=key,
        label=key,
        accent="#123456",
        metrics={"mae": "lower"},
        primary_metric="mae",
    )(spec)


def _custom(tmp_path: Path) -> dict[str, Any]:
    return {
        "name": "q_custom",
        "data": {"base_dir": str(tmp_path)},
        "output": {"models_dir": str(tmp_path / "models")},
    }


# ── the snapshot names the boundary ───────────────────────────────────────────


class TestStopAtInTheSnapshot:
    def _submitted_stop_at(
        self, client: TestClient, path: str, body: dict[str, Any]
    ) -> Any:
        resp = client.post(path, json=body)
        assert resp.status_code == 200, resp.text
        run_id = resp.json()["run_id"]
        pending = client.get("/api/queue").json()["pending"]
        return next(p for p in pending if p["run_id"] == run_id)["stop_at"]

    @pytest.mark.parametrize(
        ("block", "extra", "expected"),
        [
            ("classification", {}, "epoch"),
            (
                "transfer_learning",
                {"transfer_learning": {"mode": "feature_extraction"}},
                "epoch",
            ),
            (
                "grid_search",
                {"grid_search": {"hyperparameters": {"training.epochs": [1, 2]}}},
                "trial",
            ),
            (
                "random_search",
                {
                    "random_search": {
                        "n_trials": 2,
                        "search_space": {
                            "training.learning_rate": {
                                "type": "uniform",
                                "low": 0.001,
                                "high": 0.01,
                            }
                        },
                    }
                },
                "trial",
            ),
            ("cross_validation", {"cross_validation": {"n_folds": 2}}, "fold"),
            (
                "model_comparison",
                {"model_comparison": {"model_names": ["resnet18", "resnet34"]}},
                "model",
            ),
        ],
    )
    def test_classification_blocks(
        self,
        client_and_routes: tuple,
        tmp_path: Path,
        block: str,
        extra: dict[str, Any],
        expected: str,
    ) -> None:
        client, _ = client_and_routes
        body = _classification(tmp_path, block=block, **extra)

        assert self._submitted_stop_at(client, "/api/experiment/run", body) == expected

    @pytest.mark.parametrize(
        ("path", "wrap", "expected"),
        [
            ("/api/regression/run", None, "epoch"),
            ("/api/regression/cv", {"n_folds": 2}, "fold"),
            (
                "/api/regression/compare",
                {"model_names": ["resnet18", "resnet34"]},
                "model",
            ),
            (
                "/api/regression/sweep",
                {"search_space": {"training.learning_rate": [0.01, 0.1]}},
                "trial",
            ),
            (
                "/api/regression/sweep",
                {
                    "mode": "random",
                    "n_trials": 2,
                    "search_space": {
                        "training.learning_rate": {
                            "type": "uniform",
                            "low": 0.001,
                            "high": 0.01,
                        }
                    },
                },
                "trial",
            ),
            ("/api/regression/replicates", {"n_replicates": 2}, "replicate"),
            (
                "/api/regression/replicated-comparison",
                {"variants": {"a": {}, "b": {"training.learning_rate": 0.1}}},
                "replicate",
            ),
        ],
    )
    def test_standalone_strategies(
        self,
        client_and_routes: tuple,
        tmp_path: Path,
        path: str,
        wrap: dict[str, Any] | None,
        expected: str,
    ) -> None:
        client, _ = client_and_routes
        config = _regression(tmp_path)
        body = config if wrap is None else {"config": config, **wrap}

        assert self._submitted_stop_at(client, path, body) == expected

    @pytest.mark.parametrize(
        ("model", "expected"), [("autoencoder", "epoch"), ("patchcore", "phase")]
    )
    def test_anomaly_depends_on_the_model(
        self, client_and_routes: tuple, tmp_path: Path, model: str, expected: str
    ) -> None:
        client, _ = client_and_routes

        stop_at = self._submitted_stop_at(
            client, "/api/anomaly/run", _anomaly(tmp_path, model)
        )

        assert stop_at == expected

    @pytest.mark.parametrize(
        ("owns_loop", "expected"), [(False, "epoch"), (True, None)]
    )
    def test_custom_task_depends_on_who_owns_the_loop(
        self,
        client_and_routes: tuple,
        tmp_path: Path,
        owns_loop: bool,
        expected: str | None,
    ) -> None:
        client, _ = client_and_routes
        _register_custom("qtoy", owns_loop=owns_loop)

        stop_at = self._submitted_stop_at(
            client, "/api/custom/qtoy/run", _custom(tmp_path)
        )

        assert stop_at == expected

    def test_custom_sweeps_and_replicates_still_stop_between_units(
        self, client_and_routes: tuple, tmp_path: Path
    ) -> None:
        """Even a task that owns its loop ends between trials and replicates."""
        client, _ = client_and_routes
        _register_custom("qtoy2", owns_loop=True)
        config = _custom(tmp_path)

        sweep = self._submitted_stop_at(
            client,
            "/api/custom/qtoy2/sweep",
            {"config": config, "search_space": {"training.epochs": [1, 2]}},
        )
        replicates = self._submitted_stop_at(
            client,
            "/api/custom/qtoy2/replicates",
            {"config": config, "n_replicates": 2},
        )

        assert (sweep, replicates) == ("trial", "replicate")

    def test_the_active_job_carries_it_too(self, client_and_routes: tuple) -> None:
        client, routes_mod = client_and_routes
        routes_mod._RUN_QUEUE._active.stop_at = "fold"

        assert client.get("/api/queue").json()["active"]["stop_at"] == "fold"


# ── DELETE refuses a stop it cannot deliver ───────────────────────────────────


class TestCancellingAJobThatCannotStop:
    def test_running_job_with_no_stop_point_is_a_409(
        self, client_and_routes: tuple
    ) -> None:
        client, routes_mod = client_and_routes
        active = routes_mod._RUN_QUEUE._active
        active.stop_at = None

        resp = client.delete("/api/queue/holding")

        assert resp.status_code == 409
        assert "Nada foi interrompido" in resp.json()["detail"]
        # Nothing claims a stop: the token is untouched.
        assert active.cancel_token.cancelled is False

    def test_a_pending_job_is_still_dropped(
        self, client_and_routes: tuple, tmp_path: Path
    ) -> None:
        client, _ = client_and_routes
        _register_custom("qtoy3", owns_loop=True)
        run_id = client.post("/api/custom/qtoy3/run", json=_custom(tmp_path)).json()[
            "run_id"
        ]

        resp = client.delete(f"/api/queue/{run_id}")

        assert resp.status_code == 200
        assert resp.json()["status"] == "cancelled"
        assert client.get("/api/queue").json()["pending"] == []

    def test_running_job_that_can_stop_is_still_asked_to(
        self, client_and_routes: tuple
    ) -> None:
        client, routes_mod = client_and_routes
        routes_mod._RUN_QUEUE._active.stop_at = "replicate"

        assert client.delete("/api/queue/holding").status_code == 200
        assert routes_mod._RUN_QUEUE._active.cancel_token.cancelled is True


class TestQueueUnit:
    def test_cancel_raises_for_a_running_job_with_no_stop_point(self) -> None:
        async def scenario() -> QueuedJob:
            queue = RunQueue(on_start=lambda job: None, on_finish=lambda job: None)
            gate = asyncio.Event()

            async def blocking() -> None:
                await gate.wait()

            job = QueuedJob(
                run_id="a",
                label="a",
                task="custom:x",
                strategy="simple",
                start=blocking,
                stop_at=None,
            )
            queue.submit(job)
            await asyncio.sleep(0.01)
            try:
                with pytest.raises(NotStoppableError):
                    queue.cancel("a")
                assert queue.snapshot()["active"]["stop_at"] is None
            finally:
                gate.set()
                await asyncio.sleep(0.02)
            return job

        assert asyncio.run(scenario()).cancel_token.cancelled is False


# ── every executor hands the token on ─────────────────────────────────────────


def _run_executor(routes_mod: Any, coro: Any, token: CancellationToken) -> dict:
    routes_mod._active_cancel_token = token
    routes_mod._event_queue = None
    asyncio.run(coro)
    return dict(routes_mod._current_run)


class _Runner:
    _cancel_token: CancellationToken | None = None

    def primary_metric(self) -> str:
        return "r2"


class TestExecutorsHandTheTokenOn:
    def test_comparison(self, client_and_routes: tuple, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        _, routes_mod = client_and_routes
        token = CancellationToken()
        runner = _Runner()

        def fake(runner_arg, config, names, metric):  # type: ignore[no-untyped-def]
            assert runner_arg._cancel_token is token
            token.cancel()  # pressed during the second model
            return [
                ComparisonTrial("a", "success", {"r2": 0.9}),
                ComparisonTrial("b", STOPPED, {"r2": 0.1}),
            ]

        monkeypatch.setattr(routes_mod, "run_model_comparison", fake)
        state = _run_executor(
            routes_mod,
            routes_mod._execute_comparison(
                runner, {"name": "c"}, ["a", "b"], "r2", "r"
            ),
            token,
        )

        assert state["status"] == "completed"
        assert state["report"]["stopped"] is True
        assert state["report"]["stopped_count"] == 1
        assert state["report"]["failed_count"] == 0
        assert [t["model_arch"] for t in state["report"]["top_3"]] == ["a"]

    def test_sweep(self, client_and_routes: tuple, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        from visionforge.gui.api.schemas import SweepRequest

        _, routes_mod = client_and_routes
        token = CancellationToken()
        runner = _Runner()
        seen: dict[str, Any] = {}

        def fake(runner_arg, base, space, **kwargs):  # type: ignore[no-untyped-def]
            seen["token"] = runner_arg._cancel_token
            return [SweepTrial(0, {}, "success", {"r2": 0.5})]

        monkeypatch.setattr(routes_mod, "run_sweep", fake)
        req = SweepRequest(config={"name": "s"}, search_space={"x": [1, 2, 3]})
        state = _run_executor(
            routes_mod,
            routes_mod._execute_sweep(runner, {"name": "s"}, req, "r2", "r"),
            token,
        )

        assert seen["token"] is token
        # The table needs it: total_trials counts trials that ran.
        assert state["report"]["planned_trials"] == 3

    def test_cv(self, client_and_routes: tuple) -> None:
        from visionforge.blocks.regression_cv import CrossValidationReport, FoldResult
        from visionforge.gui.api.schemas import TaskCvRequest

        _, routes_mod = client_and_routes
        token = CancellationToken()
        seen: dict[str, Any] = {}

        def fake_cv(config, **kwargs):  # type: ignore[no-untyped-def]
            seen["token"] = kwargs["cancel_token"]
            return CrossValidationReport(
                n_folds=3,
                metric="r2",
                folds=[FoldResult(0, "success", 8, 2, {"r2": 0.5})],
                aggregate={"r2": {"mean": 0.5, "std": None, "n": 1}},
            )

        req = TaskCvRequest(config={"name": "cv"}, n_folds=3)
        state = _run_executor(
            routes_mod,
            routes_mod._execute_task_cv(object(), req, "r", fake_cv, "Regression"),
            token,
        )

        assert seen["token"] is token
        assert state["status"] == "completed"

    def test_a_replicate_that_finished_during_the_stop_completes_the_job(
        self, client_and_routes: tuple, tmp_path: Path
    ) -> None:
        """The ADR-111 review's reproduction, through the real orchestrator.

        A custom task that owns its loop runs its replicate to the end whatever
        the token says. Reading "stopped" off the token labelled that finished
        replicate cut and failed the job with "stopped before the first
        replicate finished" -- with three seeds asked and one done.
        """
        _, routes_mod = client_and_routes
        token = CancellationToken()

        class _OwnsItsLoop:
            _cancel_token: CancellationToken | None = None

            class config_type:  # noqa: N801 - the protocol's attribute name
                @staticmethod
                def model_validate(d: dict[str, Any]) -> dict[str, Any]:
                    return d

            def run(self, cfg: Any) -> RunResult:
                token.cancel()  # pressed during seed 1, which still finishes
                return RunResult(metrics={"r2": 0.5}, status="success")

            def metrics(self, result: RunResult) -> dict[str, float]:
                return dict(result.metrics)

            def primary_metric(self) -> str:
                return "r2"

        state = _run_executor(
            routes_mod,
            routes_mod._execute_replicates(
                _OwnsItsLoop(),
                {"name": "rep", "output": {"reports_dir": str(tmp_path)}},
                [1, 2, 3],
                "r2",
                "r",
            ),
            token,
        )

        assert state["status"] == "completed"
        report = state["report"]
        assert report["total_replicates"] == 1
        assert report["successful_replicates"] == 1
        # Nothing was cut, but two seeds were left unrun: the job was stopped.
        assert report["stopped"] is True

    def test_a_stop_during_the_last_unit_that_finishes_cut_nothing(
        self, client_and_routes: tuple, tmp_path: Path
    ) -> None:
        client, routes_mod = client_and_routes
        token = CancellationToken()
        calls: list[int] = []

        class _LastSeedFinishes:
            _cancel_token: CancellationToken | None = None

            class config_type:  # noqa: N801 - the protocol's attribute name
                @staticmethod
                def model_validate(d: dict[str, Any]) -> dict[str, Any]:
                    return d

            def run(self, cfg: Any) -> RunResult:
                calls.append(1)
                if len(calls) == 2:
                    token.cancel()  # pressed during the last seed, which finishes
                return RunResult(metrics={"r2": 0.5}, status="success")

            def metrics(self, result: RunResult) -> dict[str, float]:
                return dict(result.metrics)

            def primary_metric(self) -> str:
                return "r2"

        state = _run_executor(
            routes_mod,
            routes_mod._execute_replicates(
                _LastSeedFinishes(),
                {"name": "rep", "output": {"reports_dir": str(tmp_path)}},
                [1, 2],
                "r2",
                "r",
            ),
            token,
        )

        assert state["report"]["stopped"] is False
        assert client.get("/api/experiment/result/r").json()["stopped"] is False

    @pytest.mark.parametrize(
        ("report", "expected"),
        [
            ({"train": {"total_epochs": 1, "stopped": True}}, True),
            ({"train": {"total_epochs": 3, "stopped": False}}, False),
            ({"detection": {"stopped": True}}, True),
            ({"stopped": True, "trials": []}, True),
            ({"task": "custom:x", "stopped": False}, False),
            ({}, False),
        ],
    )
    def test_single_runs_carry_the_marker_too(
        self, client_and_routes: tuple, report: dict[str, Any], expected: bool
    ) -> None:
        client, routes_mod = client_and_routes
        routes_mod._current_run = {
            "run_id": "single",
            "status": "completed",
            "error": None,
            "report": report,
            "run_dir": None,
        }

        assert client.get("/api/experiment/result/single").json()["stopped"] is expected

    def test_stopped_before_any_replicate_finished_is_not_a_failure(
        self, client_and_routes: tuple, monkeypatch, tmp_path: Path
    ) -> None:  # type: ignore[no-untyped-def]
        """A stop the researcher asked for is served as a result, never a 500.

        The first version failed the job ("stopped before the first replicate
        finished"), and the result endpoint serves a failed job as a 500.
        """
        client, routes_mod = client_and_routes
        token = CancellationToken()
        token.cancel()
        runner = _Runner()

        def fake(runner_arg, base, seeds, metric, progress_callback=None):  # type: ignore[no-untyped-def]
            assert runner_arg._cancel_token is token
            return [ReplicateTrial(seeds[0], STOPPED)]

        monkeypatch.setattr(routes_mod, "run_replicates", fake)
        state = _run_executor(
            routes_mod,
            routes_mod._execute_replicates(
                runner,
                {"name": "rep", "output": {"reports_dir": str(tmp_path)}},
                [1, 2, 3],
                "r2",
                "r",
            ),
            token,
        )

        assert state["status"] == "completed"
        resp = client.get("/api/experiment/result/r")
        assert resp.status_code == 200
        assert resp.json()["stopped"] is True  # the run-level marker
        report = resp.json()["report"]
        assert report["stopped"] is True
        assert report["successful_replicates"] == 0
        assert report["headline"] is None

    def test_replicated_comparison_keeps_what_ran(
        self, client_and_routes: tuple, monkeypatch
    ) -> None:  # type: ignore[no-untyped-def]
        """No pair left to test is a failure for a finished run, not a stopped one."""
        from visionforge.gui.api.schemas import ReplicatedComparisonRequest

        _, routes_mod = client_and_routes
        token = CancellationToken()
        token.cancel()
        runner = _Runner()

        def fake(runner_arg, base, variants, seeds, metric, **kwargs):  # type: ignore[no-untyped-def]
            assert runner_arg._cancel_token is token
            return {
                "variants": {
                    "a": {"successful": 2, "trials": [{"status": "success"}] * 2}
                },
                "not_run": ["b"],
                "comparisons": [],
                "skipped_variants": ["a"],
            }

        monkeypatch.setattr(routes_mod, "run_replicated_comparison", fake)
        req = ReplicatedComparisonRequest(
            config={"name": "rc"}, variants={"a": {}, "b": {}}
        )
        state = _run_executor(
            routes_mod,
            routes_mod._execute_replicated_comparison(
                runner, {"name": "rc"}, req, [1, 2], "r2", "r"
            ),
            token,
        )

        assert state["status"] == "completed"
        assert state["report"]["stopped"] is True  # variant b never ran

    def test_custom_task(self, client_and_routes: tuple, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        from types import SimpleNamespace

        _, routes_mod = client_and_routes
        token = CancellationToken()
        seen: dict[str, Any] = {}

        class _FakeEngine:
            def __init__(self, info: Any, cfg: Any) -> None:
                pass

            def run(
                self, progress_callback: Any = None, cancel_token: Any = None
            ) -> Any:
                seen["token"] = cancel_token
                return SimpleNamespace(
                    metrics={},
                    best_epoch=0,
                    total_epochs=0,
                    device_used="cpu",
                    run_dir=Path("."),
                    stopped=True,
                )

        monkeypatch.setattr(routes_mod, "GenericTaskEngine", _FakeEngine)
        info = SimpleNamespace(key="toy")
        state = _run_executor(
            routes_mod,
            routes_mod._execute_custom_task(info, BaseTaskConfig, "r"),
            token,
        )

        assert seen["token"] is token
        assert state["status"] == "completed"
        assert state["report"]["stopped"] is True
