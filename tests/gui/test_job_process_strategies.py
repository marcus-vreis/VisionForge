"""The strategies of ``strategies.py`` in a server job's child process (ADR-252).

Model comparison, sweep, replicates and paired comparison run their real
orchestrators (``run_model_comparison``, ``run_sweep``, ``run_replicates``,
``run_replicated_comparison``) over a stand-in runner whose ``run`` trains
nothing; the task K-fold runs a stand-in for its front. Each kind goes through
its ``strategies._execute_*`` once in a real child Python and once in the
training thread, and the two are compared: the events, the run dict and its
report, and every file written (the units' folders, the summaries, the
History group). The stand-ins live in a module written to ``tmp_path`` that the
child imports by name, as it would the real ones.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import re
import textwrap
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import visionforge
from visionforge.gui.api import execution, job_call, strategies
from visionforge.gui.api.run_queue import QueuedJob
from visionforge.gui.api.schemas import (
    ReplicatedComparisonRequest,
    SweepRequest,
    TaskCvRequest,
)
from visionforge.utils.messages import tr

FAKE_MODULE = "vf_fake_strategy"
SRC = Path(visionforge.__file__).resolve().parents[1]
# Each kind by the strategy it is queued under (its stop point in _STOP_POINTS).
KINDS = ("comparison", "sweep", "cv", "replicates", "replicated-comparison")
RUN_ID = "grp_job"
# The last event of each kind's stream: the real run_sweep sends no closing end.
LAST = {kind: "trial_end" if kind == "sweep" else "end" for kind in KINDS}

_FAKE_SOURCE = textwrap.dedent(
    '''
    """A runner and a K-fold front that stand in for the real ones in a child."""
    import json
    import os
    import time
    from pathlib import Path
    from typing import Any

    from pydantic import BaseModel


    class FakeConfig(BaseModel):
        name: str = "grp"
        out: Path = Path("out")
        mode: str = "good"
        model: dict[str, Any] = {}
        training: dict[str, Any] = {}
        output: dict[str, str] = {}


    def _wait_for_stop(token):
        """Hold the first unit until the job's stop reaches this process."""
        deadline = time.monotonic() + 60
        while not (token is not None and token.cancelled):
            if time.monotonic() > deadline:
                raise TimeoutError("the stop never arrived")
            time.sleep(0.02)


    def _unit(config, label, token, emit):
        """One unit: a folder with its run.json; None when the stop cut it."""
        out = config.out.absolute()
        pids = out / "pids"
        pids.mkdir(parents=True, exist_ok=True)
        (pids / str(os.getpid())).write_text("", encoding="utf-8")
        first = not any(out.glob("*/run.json"))
        if config.mode == "crash" and not first:
            os._exit(3)
        folder = out / label
        folder.mkdir(parents=True, exist_ok=True)
        emit({"event": "epoch_end", "epoch": 1})
        (folder / "run.json").write_text(json.dumps({"unit": label}), encoding="utf-8")
        if config.mode == "wait_stop" and first:
            _wait_for_stop(token)
            return folder, True
        return folder, False


    class FakeRunner:
        """Trains nothing: each run leaves a folder and a score from its name."""

        config_type = FakeConfig
        _cancel_token = None
        _progress_callback = None

        def primary_metric(self):
            return "score"

        def metrics(self, result):
            return dict(result.metrics)

        def run(self, cfg):
            from visionforge.core.task_runner import RunResult

            def emit(event):
                if self._progress_callback is not None:
                    self._progress_callback(event)

            label = f"{cfg.name}-{cfg.model.get('name', '')}"
            folder, cut = _unit(cfg, label, self._cancel_token, emit)
            emit({"event": "end", "total_epochs": 1})
            score = round(0.1 + 0.01 * (sum(map(ord, label)) % 50), 4)
            return RunResult(
                metrics={"score": score},
                status="success",
                training_time_s=1.0,
                stopped=cut,
                run_dir=folder,
                selection_metrics={"score": score},
            )


    def fake_kfold(
        config, *, n_folds, shuffle, seed, select_on, progress_callback, cancel_token
    ):
        """Stands in for a task's K-fold front: one unit per fold."""
        from visionforge.blocks._cv_common import CrossValidationReport, FoldResult

        if config.mode == "error":
            raise ValueError(f"no rows to split into {n_folds} folds")
        folds = []
        for fold in range(n_folds):
            progress_callback(
                {"event": "trial_start", "trial_index": fold, "total_trials": n_folds}
            )
            folder, cut = _unit(config, f"fold{fold}", cancel_token, progress_callback)
            status = "stopped" if cut else "success"
            folds.append(
                FoldResult(fold, status, 8, 2, {"score": 0.5 + fold / 10}, "", str(folder))
            )
            progress_callback({"event": "trial_end", "trial_index": fold, "status": status})
            if cancel_token.cancelled:
                break
        progress_callback({"event": "end", "total_epochs": 0, "total_trials": len(folds)})
        done = [f.metrics["score"] for f in folds if f.status == "success"]
        mean = sum(done) / len(done) if done else None
        return CrossValidationReport(
            n_folds=n_folds,
            metric="score",
            folds=folds,
            aggregate={"score": {"mean": mean, "std": None, "n": len(done)}},
            select_on=select_on,
        )
    '''
)


@pytest.fixture(scope="module")
def fake_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    folder = Path(tmp_path_factory.mktemp("fake_strategy"))
    (folder / f"{FAKE_MODULE}.py").write_text(_FAKE_SOURCE, encoding="utf-8")
    return folder


@pytest.fixture
def fake(fake_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """The stand-ins, importable here and in the child; no job states left over."""
    monkeypatch.syspath_prepend(str(fake_dir))
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(SRC), str(fake_dir)]))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(execution, "_JOB_STATES", {})
    monkeypatch.setattr(execution, "_latest_run_id", None)
    module = importlib.import_module(FAKE_MODULE)
    monkeypatch.setattr(
        strategies, "run_regression_cross_validation", module.fake_kfold
    )
    return module


def _config(mode: str) -> dict[str, Any]:
    """The group's config dict, its folders relative to the server's folder.

    The units train inside the models folder, as real runs do: History refuses
    a group whose runs lie outside it.
    """
    return {
        "name": "grp",
        "out": "models",
        "mode": mode,
        "model": {"name": "m0"},
        "training": {"lr": 0.01},
        "output": {"reports_dir": "reports", "models_dir": "models"},
    }


def _job(kind: str, fake: ModuleType, mode: str) -> Any:
    """The ``_execute_*`` coroutine of ``kind`` as ``_start_*`` would queue it."""
    runner = fake.FakeRunner()
    config = _config(mode)
    models = Path("models").absolute()
    if kind == "comparison":
        return strategies._execute_comparison(
            runner, config, ["m1", "m2", "m3"], "score", RUN_ID, models_dir=models
        )
    if kind == "sweep":
        space = {"training.lr": [0.1, 0.2, 0.3]}
        sweep = SweepRequest(config=config, search_space=space)
        if mode == "error":
            # An unknown mode makes the real run_sweep raise before any trial.
            sweep = sweep.model_copy(update={"mode": "bogus"})
        return strategies._execute_sweep(
            runner, config, sweep, "score", RUN_ID, models_dir=models
        )
    if kind == "cv":
        req = TaskCvRequest(config=config, n_folds=3)
        return strategies._execute_task_cv(
            fake.FakeConfig.model_validate(config),
            req,
            RUN_ID,
            strategies.run_regression_cross_validation,
            "Regression",
            models_dir=models,
        )
    if kind == "replicates":
        return strategies._execute_replicates(
            runner, config, [1, 2, 3], "score", RUN_ID, models_dir=models
        )
    variants = {"a": {"training.lr": 0.1}, "b": {"training.lr": 0.2}}
    req_rc = ReplicatedComparisonRequest(config=config, variants=variants, seeds=[1, 2])
    return strategies._execute_replicated_comparison(
        runner, config, req_rc, [1, 2], "score", RUN_ID, models_dir=models
    )


def _start(kind: str, *, isolated: bool) -> execution.JobState:
    async def _nothing() -> None:
        return None

    job = QueuedJob(
        run_id=RUN_ID,
        label=RUN_ID,
        task="regression",
        strategy=kind,
        start=_nothing,
        isolated=isolated,
    )
    execution._begin_job(job)
    return execution._job_state(RUN_ID)


def _run(
    kind: str,
    fake: ModuleType,
    *,
    isolated: bool,
    mode: str = "good",
    stop_on_first_event: bool = False,
) -> tuple[execution.JobState, list[dict[str, Any]]]:
    """Run one job of ``kind`` to its end; its state and the events its log received."""
    execution._JOB_STATES.pop(RUN_ID, None)
    state = _start(kind, isolated=isolated)

    async def read() -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        async for event in state.log.read():
            if stop_on_first_event and not events:
                state.token.cancel()  # what DELETE /api/queue/{id} does
            events.append(event)
        return events

    async def scenario() -> list[dict[str, Any]]:
        reader = asyncio.create_task(read())
        await asyncio.wait_for(_job(kind, fake, mode), 180)
        return await asyncio.wait_for(reader, 10)

    return state, asyncio.run(scenario())


_STAMPS = (
    re.compile(r"\d{8}_\d{6}_\d{6}"),
    re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?"),
)


def _neutral(text: str, root: Path) -> str:
    """``text`` with this root's path and the write times taken out."""
    for form in (json.dumps(str(root))[1:-1], str(root), root.as_posix()):
        text = text.replace(form, "<root>")
    for stamp in _STAMPS:
        text = stamp.sub("<time>", text)
    return text


def _pids(root: Path) -> set[int]:
    return {int(p.name) for p in (root / "models" / "pids").iterdir()}


def _written(root: Path) -> dict[str, str]:
    """Every file under ``root`` but the pid marks, path and text made neutral."""
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if not path.is_file() or rel.startswith("models/pids/"):
            continue
        text = "<binary>" if path.suffix == ".png" else path.read_text("utf-8")
        files[_neutral(rel, root)] = _neutral(text, root)
    return files


def _run_dict(state: execution.JobState, root: Path) -> str:
    return _neutral(json.dumps(state.run, sort_keys=True), root)


def _in(root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root.mkdir()
    monkeypatch.chdir(root)
    return root


@pytest.mark.parametrize("kind", KINDS)
def test_a_server_strategy_runs_in_a_child_as_the_thread_would(
    kind: str, fake: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    thread_root = _in(tmp_path / "thread", monkeypatch)
    thread, thread_events = _run(kind, fake, isolated=False)
    child_root = _in(tmp_path / "child", monkeypatch)
    child, child_events = _run(kind, fake, isolated=True)

    assert child.run["status"] == "completed", child.run["error"]
    assert thread.run["status"] == "completed", thread.run["error"]
    assert _pids(thread_root) == {os.getpid()}
    (child_pid,) = _pids(child_root)
    assert child_pid != os.getpid()
    units = [e["trial_index"] for e in child_events if e["event"] == "trial_start"]
    assert units == list(range(len(units)))
    assert len(units) >= 3
    assert child_events[-1]["event"] == LAST[kind]
    assert child_events == thread_events
    # A group job names no one run folder, in the child as in the thread (ADR-249).
    assert child.run["run_dir"] is None
    assert _run_dict(child, child_root) == _run_dict(thread, thread_root)
    child_files = _written(child_root)
    assert child_files == _written(thread_root)
    assert any(name.startswith("models/") for name in child_files)
    assert any(name.startswith("reports/") for name in child_files)


def test_a_paired_comparison_keeps_its_int_seed_keys_across_the_child(
    fake: ModuleType,
) -> None:
    state, _ = _run("replicated-comparison", fake, isolated=True)
    assert state.run["status"] == "completed", state.run["error"]
    per_seed = state.run["report"]["variants"]["a"]["per_seed"]
    assert sorted(per_seed) == [1, 2]


@pytest.mark.parametrize("kind", KINDS)
def test_a_stop_ends_the_child_strategy_at_its_stop_point(
    kind: str, fake: ModuleType
) -> None:
    assert execution._stop_point(kind) in {"model", "trial", "fold", "replicate"}
    state, events = _run(
        kind, fake, isolated=True, mode="wait_stop", stop_on_first_event=True
    )

    assert state.run["status"] == "completed", state.run["error"]
    assert state.run["report"]["stopped"] is True
    # The unit in flight saw the stop and ended; no other unit started.
    closing = ["end"] if LAST[kind] == "end" else []
    assert [e["event"] for e in events] == [
        "trial_start",
        "epoch_end",
        "trial_end",
        *closing,
    ]
    assert events[2]["status"] == "stopped"
    assert _pids(Path.cwd()) != {os.getpid()}


@pytest.mark.parametrize("kind", KINDS)
def test_a_child_that_dies_fails_the_strategy_and_closes_its_stream(
    kind: str, fake: ModuleType
) -> None:
    state, events = _run(kind, fake, isolated=True, mode="crash")

    assert state.run["status"] == "failed"
    assert state.run["error"] == tr("job.child_died", code=3)
    assert state.run["report"] is None
    # The first unit's events arrived before the child died; the stream closed.
    assert [e["event"] for e in events] == [
        "trial_start",
        "epoch_end",
        "trial_end",
        "trial_start",
    ]
    # Only the first unit's folder: no summary, no History group.
    assert len(list(Path("models").rglob("run.json"))) == 1
    assert not Path("reports").exists()


@pytest.mark.parametrize("kind", ["sweep", "cv"])
def test_an_error_in_the_child_is_worded_as_the_thread_words_it(
    kind: str, fake: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _in(tmp_path / "thread", monkeypatch)
    thread, _ = _run(kind, fake, isolated=False, mode="error")
    _in(tmp_path / "child", monkeypatch)
    child, events = _run(kind, fake, isolated=True, mode="error")

    assert child.run["status"] == thread.run["status"] == "failed"
    assert child.run["error"] == thread.run["error"]
    assert child.run["error"].startswith("ValueError: ")
    assert events == []


@pytest.mark.parametrize("kind", KINDS)
def test_a_local_strategy_keeps_the_training_thread(
    kind: str, fake: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def no_child(*_: Any, **__: Any) -> None:
        raise AssertionError("a local job must not start a child")

    monkeypatch.setattr(job_call, "run_in_child", no_child)
    state, events = _run(kind, fake, isolated=False)
    assert state.run["status"] == "completed", state.run["error"]
    assert _pids(Path.cwd()) == {os.getpid()}
    assert events[-1]["event"] == LAST[kind]


def test_a_job_whose_runner_cannot_cross_fails_before_a_child_starts(
    fake: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    started: list[Any] = []

    async def no_child(*args: Any, **_: Any) -> None:
        started.append(args[0])

    monkeypatch.setattr(job_call, "run_in_child", no_child)
    runner = fake.FakeRunner()
    runner.task = "custom"  # a custom task's runner holds its task
    _start("replicates", isolated=True)

    async def scenario() -> list[dict[str, Any]]:
        await strategies._execute_replicates(
            runner, _config("good"), [1, 2], "score", RUN_ID, models_dir=Path("m")
        )
        return [event async for event in execution._job_state(RUN_ID).log.read()]

    assert asyncio.run(scenario()) == []
    state = execution._job_state(RUN_ID)
    assert state.run["status"] == "failed"
    assert state.run["error"].startswith("TypeError: FakeRunner holds ['task']")
    assert started == []
