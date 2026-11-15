"""The standalone and custom tasks hand their run.json to the profile's trackers (#286)."""

from __future__ import annotations

import asyncio
import json
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import visionforge.gui.api.custom_task_routes as custom_mod
import visionforge.gui.api.execution as execution_mod
import visionforge.gui.api.training_routes as training_mod
from visionforge.gui.api.profiles import resolve_profile
from visionforge.gui.api.tracker_exports import exported, profile_trackers

from .conftest import job_run, release_queue, start_job

_RUN_ID = "r"


class _Recorder:
    """A tracker that writes down what it was sent."""

    built: list[_Recorder] = []

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.epochs: list[tuple[int, dict[str, Any]]] = []
        self.status: str | None = None
        _Recorder.built.append(self)

    def start(self, config: dict[str, Any]) -> None:
        self.config = config

    def epoch(self, epoch: int, metrics: dict[str, Any]) -> None:
        self.epochs.append((epoch, metrics))

    def finish(self, status: str) -> None:
        self.status = status


@pytest.fixture(autouse=True)
def _clean() -> Iterator[None]:
    _Recorder.built = []
    yield
    release_queue(execution_mod)


def _write_record(run_dir: Path, task: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "status": "completed",
        "stopped": False,
        "config": {"name": "exp", "task": task},
        "metrics": {"best_epoch": 1},
        "history": [{"epoch": 1, "train_loss": 0.5}],
    }
    (run_dir / "run.json").write_text(json.dumps(record), encoding="utf-8")


def _fake_block(task: str, run_dir: Path, report_key: str) -> type:
    class _Block:
        def setup(self, config: Any) -> None:
            self.config = config

        def run(self) -> None:
            _write_record(run_dir, task)

        def report(self) -> dict[str, Any]:
            return {report_key: {"run_dir": str(run_dir)}}

    return _Block


# executor -> (block module, block class, report key)
EXECUTORS = {
    "detection": ("visionforge.blocks.detection", "DetectionBlock", "detection"),
    "regression": ("visionforge.blocks.regression", "RegressionBlock", "train"),
    "segmentation": ("visionforge.blocks.segmentation", "SegmentationBlock", "train"),
    "anomaly": ("visionforge.blocks.anomaly", "AnomalyBlock", "train"),
}


def _run(coro: Any) -> dict[str, Any]:
    start_job(execution_mod, _RUN_ID)
    asyncio.run(coro)
    return dict(job_run(execution_mod, _RUN_ID))


@pytest.mark.parametrize("task", EXECUTORS)
def test_a_finished_task_run_is_exported(
    task: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module_name, class_name, report_key = EXECUTORS[task]
    run_dir = tmp_path / task
    module = types.ModuleType(module_name)
    setattr(module, class_name, _fake_block(task, run_dir, report_key))
    monkeypatch.setitem(sys.modules, module_name, module)
    executor = getattr(training_mod, f"_execute_{task}")
    config = types.SimpleNamespace(model=types.SimpleNamespace(name="m"))

    state = _run(exported(executor(config, _RUN_ID), _RUN_ID, (_Recorder,)))

    assert state["status"] == "completed"
    (tracker,) = _Recorder.built
    assert tracker.run_dir == run_dir
    assert tracker.config["task"] == task
    assert tracker.epochs == [(1, {"train_loss": 0.5})]
    assert tracker.status == "FINISHED"


def test_without_trackers_nothing_is_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module_name, class_name, report_key = EXECUTORS["regression"]
    module = types.ModuleType(module_name)
    setattr(module, class_name, _fake_block("regression", tmp_path / "r", report_key))
    monkeypatch.setitem(sys.modules, module_name, module)
    config: Any = types.SimpleNamespace(model=types.SimpleNamespace(name="m"))

    state = _run(
        exported(training_mod._execute_regression(config, _RUN_ID), _RUN_ID, ())
    )

    assert state["status"] == "completed"
    assert _Recorder.built == []


def test_a_custom_task_run_is_exported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "custom"

    class _Engine:
        def __init__(self, info: Any, cfg: Any) -> None:
            pass

        def run(self, progress: Any = None, token: Any = None) -> Any:
            _write_record(run_dir, "denoise")
            return types.SimpleNamespace(
                metrics={},
                best_epoch=1,
                total_epochs=1,
                device_used="cpu",
                run_dir=run_dir,
                stopped=False,
            )

    monkeypatch.setattr(custom_mod, "GenericTaskEngine", _Engine)
    info = types.SimpleNamespace(key="denoise")

    job = custom_mod._execute_custom_task(info, object(), _RUN_ID)
    state = _run(exported(job, _RUN_ID, (_Recorder,)))

    assert state["status"] == "completed"
    (tracker,) = _Recorder.built
    assert tracker.run_dir == run_dir
    assert tracker.status == "FINISHED"


def test_profile_trackers_follow_the_profile_settings(tmp_path: Path) -> None:
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    meta = {"settings": {"mlflow": {"enabled": True}, "wandb": {"enabled": False}}}
    (profiles / "default.json").write_text(json.dumps(meta), encoding="utf-8")
    profile = resolve_profile(None, profiles, tmp_path / "models")

    factories = profile_trackers(profile, tmp_path)

    assert [f.func.__name__ for f in factories] == ["MlflowTracker"]  # type: ignore[attr-defined]
    assert factories[0].args[0] == tmp_path / "mlflow"  # type: ignore[attr-defined]


def test_a_failed_run_is_not_exported(monkeypatch: pytest.MonkeyPatch) -> None:
    module_name, class_name, _ = EXECUTORS["anomaly"]

    class _Broken:
        def setup(self, config: Any) -> None:
            pass

        def run(self) -> None:
            raise RuntimeError("boom")

    module = types.ModuleType(module_name)
    setattr(module, class_name, _Broken)
    monkeypatch.setitem(sys.modules, module_name, module)
    config: Any = types.SimpleNamespace(model=types.SimpleNamespace(name="m"))

    job = training_mod._execute_anomaly(config, _RUN_ID)
    state = _run(exported(job, _RUN_ID, (_Recorder,)))

    assert state["status"] == "failed"
    assert _Recorder.built == []
