"""Classification's blocks of several models in a server job's child process (ADR-251).

K-fold, grid search, random search and model comparison train several models
in one job. These tests run each kind through ``execution._execute_experiment``
in a real child Python and in the training thread, and compare the two: the
events, the report, the run folder the run dict names, the files written. The
fake blocks subclass the real ones, so ``_wire_block`` treats them as it treats
the real kinds (event pump yes, trackers no); only setup, run and report are
stand-ins that train nothing.
"""

from __future__ import annotations

import asyncio
import importlib
import os
import textwrap
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import visionforge
from visionforge.gui.api import execution
from visionforge.gui.api.run_queue import QueuedJob
from visionforge.utils.config import ExperimentConfig
from visionforge.utils.messages import tr

FAKE_MODULE = "vf_fake_multi_block"
SRC = Path(visionforge.__file__).resolve().parents[1]
KINDS = ("cross_validation", "grid_search", "random_search", "model_comparison")

_FAKE_SOURCE = textwrap.dedent(
    '''
    """Blocks of several models that stand in for the real ones in a child."""
    import json
    import os
    import time
    from pathlib import Path
    from types import SimpleNamespace

    from pydantic import BaseModel

    from visionforge.blocks.cross_validation import CrossValidationBlock
    from visionforge.blocks.grid_search import GridSearchBlock
    from visionforge.blocks.model_comparison import ModelComparisonBlock
    from visionforge.blocks.random_search import RandomSearchBlock


    class FakeConfig(BaseModel):
        name: str = "multi"
        block: str = "grid_search"
        mode: str = "good"
        units: int = 3
        epochs: int = 2
        pause: float = 0.0
        out: Path = Path("out")


    class _Units:
        """Trains ``units`` models: a folder and a run.json each, one for the group."""

        def setup(self, config):
            self.config = config
            self._progress_callback = None
            self._units = []

        def _emit(self, event):
            if self._progress_callback is not None:
                self._progress_callback(event)

        def run(self):
            c = self.config
            root = c.out.absolute() / c.name
            for unit in range(c.units):
                if c.mode == "crash" and unit == 1:
                    os._exit(3)
                self._emit(
                    {"event": "trial_start", "trial_index": unit, "total_trials": c.units}
                )
                folder = root / f"unit{unit}"
                folder.mkdir(parents=True, exist_ok=True)
                for epoch in range(1, c.epochs + 1):
                    self._emit({"event": "epoch_end", "epoch": epoch, "trial_index": unit})
                    time.sleep(c.pause)
                (folder / "run.json").write_text(json.dumps({"unit": unit}))
                self._units.append(
                    {"index": unit, "run_dir": str(folder), "loss": 1 / (unit + 1)}
                )
                self._emit({"event": "trial_end", "trial_index": unit})
                # The real blocks stop between their units (ADR-111).
                if self._cancel_token is not None and self._cancel_token.cancelled:
                    break
            (root / "run.json").write_text(json.dumps({"units": len(self._units)}))
            # Like a block that keeps its last model's result: still no one folder.
            self._train_result = SimpleNamespace(model_path=folder / "best.pt")
            self._emit({"event": "end", "total_epochs": 0, "total_trials": len(self._units)})

        def report(self):
            return {
                "units": self._units,
                "stopped": len(self._units) < self.config.units,
                "pid": os.getpid(),
            }


    class FakeKFold(_Units, CrossValidationBlock):
        pass


    class FakeGrid(_Units, GridSearchBlock):
        pass


    class FakeRandom(_Units, RandomSearchBlock):
        pass


    class FakeComparison(_Units, ModelComparisonBlock):
        pass


    BY_KIND = {
        "cross_validation": FakeKFold,
        "grid_search": FakeGrid,
        "random_search": FakeRandom,
        "model_comparison": FakeComparison,
    }
    '''
)


@pytest.fixture(scope="module")
def fake_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    folder = Path(tmp_path_factory.mktemp("fake_multi_block"))
    (folder / f"{FAKE_MODULE}.py").write_text(_FAKE_SOURCE, encoding="utf-8")
    return folder


@pytest.fixture
def fake(fake_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """The fake blocks, importable here and in the child; no job states left over."""
    monkeypatch.syspath_prepend(str(fake_dir))
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(SRC), str(fake_dir)]))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(execution, "_JOB_STATES", {})
    monkeypatch.setattr(execution, "_latest_run_id", None)
    module = importlib.import_module(FAKE_MODULE)
    # "classification" is the fallback the executor looks up; never run here.
    monkeypatch.setattr(
        execution, "_block_types", lambda: {"classification": object, **module.BY_KIND}
    )
    return module


@pytest.fixture
def exported(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, Path]]:
    """What the server sent to the trackers at the end of a run."""
    sent: list[tuple[Any, Path]] = []
    monkeypatch.setattr(
        execution,
        "export_run_record",
        lambda trackers, run_dir: sent.append((trackers, run_dir)),
    )
    return sent


def _start(run_id: str, kind: str, *, isolated: bool) -> execution.JobState:
    async def _nothing() -> None:
        return None

    job = QueuedJob(
        run_id=run_id,
        label=run_id,
        task="classification",
        strategy=kind,
        start=_nothing,
        isolated=isolated,
    )
    execution._begin_job(job)
    return execution._job_state(run_id)


def _execute(
    state: execution.JobState, config: Any, *, stop_on_first_event: bool = False
) -> list[dict[str, Any]]:
    """Run the job through the executor; the events its log received, in order."""

    async def read() -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        async for event in state.log.read():
            if stop_on_first_event and not events:
                state.token.cancel()  # what DELETE /api/queue/{id} does
            events.append(event)
        return events

    async def scenario() -> list[dict[str, Any]]:
        reader = asyncio.create_task(read())
        await asyncio.wait_for(
            execution._execute_experiment(
                config,
                state.run_id,
                trackers=(object(),),  # type: ignore[arg-type]
            ),
            120,
        )
        return await asyncio.wait_for(reader, 10)

    return asyncio.run(scenario())


def _files(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _without_pid(run: dict[str, Any]) -> dict[str, Any]:
    report = {k: v for k, v in run["report"].items() if k != "pid"}
    return {**{k: v for k, v in run.items() if k != "run_id"}, "report": report}


@pytest.mark.parametrize("kind", KINDS)
def test_a_server_job_of_several_models_runs_in_a_child_as_the_thread_would(
    kind: str, fake: ModuleType, exported: list[tuple[Any, Path]]
) -> None:
    config = fake.FakeConfig(block=kind)
    child = _start(f"{kind}_child", kind, isolated=True)
    child_events = _execute(child, config)
    child_files = _files(Path("out"))

    thread = _start(f"{kind}_thread", kind, isolated=False)
    thread_events = _execute(thread, config)

    assert child.run["status"] == "completed", child.run["error"]
    assert child.run["report"]["pid"] != os.getpid()
    assert thread.run["report"]["pid"] == os.getpid()
    assert [(e["event"], e.get("trial_index")) for e in child_events] == [
        *[
            step
            for unit in range(3)
            for step in (
                ("trial_start", unit),
                ("epoch_end", unit),
                ("epoch_end", unit),
                ("trial_end", unit),
            )
        ],
        ("end", None),
    ]
    assert child_events == thread_events
    # No one run folder, and no tracker export: the thread's rules for these kinds.
    assert child.run["run_dir"] is None
    assert _without_pid(child.run) == _without_pid(thread.run)
    assert _files(Path("out")) == child_files
    assert set(child_files) == {
        "multi/run.json",
        "multi/unit0/run.json",
        "multi/unit1/run.json",
        "multi/unit2/run.json",
    }
    assert exported == []


@pytest.mark.parametrize("kind", KINDS)
def test_a_stop_ends_the_child_at_the_blocks_stop_point(
    kind: str, fake: ModuleType, exported: list[tuple[Any, Path]]
) -> None:
    assert execution._stop_point(kind) in {"fold", "trial", "model"}
    state = _start(f"{kind}_stop", kind, isolated=True)
    config = fake.FakeConfig(block=kind, units=5, epochs=3, pause=0.3)
    events = _execute(state, config, stop_on_first_event=True)

    assert state.run["status"] == "completed", state.run["error"]
    report = state.run["report"]
    assert report["stopped"] is True
    # The unit in flight ended with all its epochs; no other one started.
    assert [unit["index"] for unit in report["units"]] == [0]
    assert [e["event"] for e in events] == [
        "trial_start",
        "epoch_end",
        "epoch_end",
        "epoch_end",
        "trial_end",
        "end",
    ]
    assert Path("out", "multi", "run.json").is_file()


@pytest.mark.parametrize("kind", KINDS)
def test_a_child_that_dies_mid_job_fails_the_run_and_closes_its_stream(
    kind: str, fake: ModuleType, exported: list[tuple[Any, Path]]
) -> None:
    state = _start(f"{kind}_crash", kind, isolated=True)
    events = _execute(state, fake.FakeConfig(block=kind, mode="crash"))

    assert state.run["status"] == "failed"
    assert state.run["error"] == tr("job.child_died", code=3)
    assert state.run["report"] is None
    # The first unit's events arrived before the child died.
    assert [e["event"] for e in events] == [
        "trial_start",
        "epoch_end",
        "epoch_end",
        "trial_end",
    ]
    assert exported == []


@pytest.mark.parametrize("kind", KINDS)
def test_a_local_job_of_several_models_keeps_the_training_thread(
    kind: str, fake: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def no_child(*_: Any, **__: Any) -> None:
        raise AssertionError("a local job must not start a child")

    monkeypatch.setattr(execution, "_execute_in_child", no_child)
    state = _start(f"{kind}_local", kind, isolated=False)
    events = _execute(state, fake.FakeConfig(block=kind, units=1))
    assert state.run["status"] == "completed", state.run["error"]
    assert state.run["report"]["pid"] == os.getpid()
    assert state.run["run_dir"] is None
    assert events[-1]["event"] == "end"


@pytest.mark.parametrize("kind", KINDS)
def test_the_real_block_and_config_cross_to_the_child(
    kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real class and ``ExperimentConfig`` reach the child; setup fails there as here."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join([str(SRC), os.environ.get("PYTHONPATH", "")])
    )
    monkeypatch.setattr(execution, "_JOB_STATES", {})
    monkeypatch.setattr(execution, "_latest_run_id", None)
    # The block is named but its own section is not: setup refuses it.
    config = ExperimentConfig.model_validate(
        {
            "name": "real",
            "block": kind,
            "model": {"name": "resnet18", "num_classes": 2, "pretrained": False},
            "data": {"base_dir": str(tmp_path)},
        }
    )
    child = _start(f"{kind}_real_child", kind, isolated=True)
    _execute(child, config)
    thread = _start(f"{kind}_real_thread", kind, isolated=False)
    _execute(thread, config)

    assert child.run["status"] == thread.run["status"] == "failed"
    assert child.run["error"] == thread.run["error"]
    assert child.run["error"].startswith("ValueError: ")
    assert kind.replace("_", "") in child.run["error"].lower().replace("_", "")
