"""The operator's sandbox command in front of a job's child process (ADR-254).

The wrapper here is a Python script that writes down what it was given and then
runs the child's command with its own pipes, as ``bwrap`` or ``sudo`` would.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import textwrap
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from loguru import logger

import visionforge
from visionforge.core.cancellation import CancellationToken
from visionforge.gui.api import submission
from visionforge.gui.api.dataset_scope import DatasetScope, dataset_scope
from visionforge.gui.api.job_process import WORKER_MODULE, ChildJob, child_spec
from visionforge.gui.api.job_wrapper import (
    READ_ENV,
    WORKDIR_ENV,
    WRAPPER_ENV,
    WRITE_ENV,
    JobFolders,
    JobWrapperError,
    job_folders,
    wrapper_command,
    wrapper_env,
)
from visionforge.gui.api.profiles import Profile
from visionforge.gui.api.run_queue import QueuedJob
from visionforge.gui.server_mode import ServerMode
from visionforge.utils.messages import tr

SRC = Path(visionforge.__file__).resolve().parents[1]
BLOCK_MODULE = "vf_wrapped_job_block"

_BLOCK = textwrap.dedent(
    """
    import os
    import time

    from pydantic import BaseModel


    class Config(BaseModel):
        name: str = "wrapped"
        epochs: int = 2
        pause: float = 0.0


    class Block:
        def setup(self, config):
            self.config = config
            self._progress_callback = None
            self._cancel_token = None
            self._resume_dir = None
            self._train_result = None
            self._report = {}

        def run(self):
            epoch = 0
            while epoch < self.config.epochs and not self._cancel_token.cancelled:
                epoch += 1
                self._progress_callback({"type": "epoch_end", "epoch": epoch})
                time.sleep(self.config.pause)
            self._report = {
                "train": {"run_dir": os.getcwd(), "epochs": epoch},
                "told": {k: v for k, v in os.environ.items() if "_JOB_" in k},
            }

        def report(self):
            return self._report
    """
)

_WRAPPER = textwrap.dedent(
    """
    import json, os, subprocess, sys

    record, command = sys.argv[1], sys.argv[2:]
    told = {k: v for k, v in os.environ.items() if "_JOB_" in k}
    with open(record, "w", encoding="utf-8") as out:
        json.dump({"command": command, "told": told, "cwd": os.getcwd()}, out)
    sys.exit(subprocess.call(command))
    """
)


@pytest.fixture(autouse=True)
def no_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(WRAPPER_ENV, raising=False)


@pytest.fixture
def block(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """The fake block, importable here and in the child."""
    folder = tmp_path / "block module"
    folder.mkdir()
    (folder / f"{BLOCK_MODULE}.py").write_text(_BLOCK, encoding="utf-8")
    monkeypatch.syspath_prepend(str(folder))
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(SRC), str(folder)]))
    sys.modules.pop(BLOCK_MODULE, None)
    return __import__(BLOCK_MODULE)


@pytest.fixture
def record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A wrapper is set; the file it writes what it was given into."""
    script = tmp_path / "a wrapper.py"
    script.write_text(_WRAPPER, encoding="utf-8")
    path = tmp_path / "record.json"
    monkeypatch.setenv(WRAPPER_ENV, f'"{sys.executable}" "{script}" "{path}"')
    return path


def _run(
    block: Any, token: CancellationToken | None = None, **config: Any
) -> tuple[Any, list[dict[str, Any]]]:
    spec = child_spec(block.Block, block.Config(**config), None, report_section="train")
    events: list[dict[str, Any]] = []
    job = ChildJob(spec, events.append, token or CancellationToken())
    return job.supervise(), events


def test_no_variable_is_no_words() -> None:
    assert wrapper_command() == []
    assert wrapper_command({WRAPPER_ENV: "   "}) == []


def test_the_variable_is_split_into_words_with_quoted_paths_whole(
    tmp_path: Path,
) -> None:
    script = tmp_path / "with space.py"
    words = wrapper_command({WRAPPER_ENV: f'"{sys.executable}" "{script}" --flag'})
    assert words == [sys.executable, str(script), "--flag"]


def test_a_command_that_cannot_be_split_is_refused() -> None:
    with pytest.raises(JobWrapperError, match="not a command line"):
        wrapper_command({WRAPPER_ENV: 'bwrap "unbalanced'})


def test_a_program_this_machine_lacks_is_refused_by_name() -> None:
    with pytest.raises(JobWrapperError, match="vf-no-such-sandbox"):
        wrapper_command({WRAPPER_ENV: "vf-no-such-sandbox --die-with-parent"})


def test_the_wrapper_is_told_the_folders_of_the_job_in_this_context(
    tmp_path: Path,
) -> None:
    assert wrapper_env(tmp_path) == {
        WORKDIR_ENV: str(tmp_path),
        WRITE_ENV: "",
        READ_ENV: "",
    }
    runs, own, shared = tmp_path / "runs", tmp_path / "own", tmp_path / "shared"
    with job_folders(JobFolders(write=(runs,), read=(own, shared))):
        told = wrapper_env(tmp_path)
    assert told[WRITE_ENV] == str(runs)
    assert told[READ_ENV] == os.pathsep.join([str(own), str(shared)])
    assert wrapper_env(tmp_path)[WRITE_ENV] == ""


def test_without_a_wrapper_the_child_is_told_nothing(block: Any) -> None:
    outcome, _ = _run(block)
    assert outcome.status == "completed"
    assert outcome.report is not None
    assert outcome.report["told"] == {}


def test_a_job_starts_through_the_wrapper_which_is_told_its_folders(
    block: Any, record: Path, tmp_path: Path
) -> None:
    runs, data = tmp_path / "runs", tmp_path / "data"
    with job_folders(JobFolders(write=(runs,), read=(data,))):
        outcome, events = _run(block)
    assert outcome.status == "completed"
    assert [event["epoch"] for event in events] == [1, 2]
    seen = json.loads(record.read_text(encoding="utf-8"))
    assert seen["command"] == [sys.executable, "-m", WORKER_MODULE]
    assert seen["told"][WRITE_ENV] == str(runs)
    assert seen["told"][READ_ENV] == str(data)
    # The working folder the wrapper is told is the one the job starts in.
    assert Path(seen["told"][WORKDIR_ENV]) == Path(seen["cwd"])
    assert outcome.report is not None
    assert Path(outcome.report["train"]["run_dir"]) == Path(seen["cwd"])


def test_a_stop_reaches_the_child_behind_the_wrapper(block: Any, record: Path) -> None:
    token = CancellationToken()
    spec = child_spec(
        block.Block, block.Config(epochs=500, pause=0.05), None, report_section="train"
    )
    first = threading.Event()
    job = ChildJob(spec, lambda _event: first.set(), token)
    result: list[Any] = []
    worker = threading.Thread(target=lambda: result.append(job.supervise()))
    worker.start()
    assert first.wait(60)
    token.cancel()
    worker.join(60)
    (outcome,) = result
    assert outcome.status == "completed"
    assert outcome.report["train"]["epochs"] < 500


def test_a_wrapper_that_is_gone_fails_the_run_and_not_the_server(
    block: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(WRAPPER_ENV, "vf-no-such-sandbox")
    outcome, events = _run(block)
    assert outcome.status == "failed"
    assert outcome.error is not None
    assert "vf-no-such-sandbox" in outcome.error
    assert outcome.error.startswith(tr("job.child_start", error="")[:10])
    assert events == []


def test_a_submitted_job_carries_the_persons_folders_to_its_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queued: list[QueuedJob] = []

    class _Lanes:
        def count_of(self, profile: str) -> int:
            return 0

        def submit(self, job: QueuedJob, lane: str, fair: bool = False) -> str:
            queued.append(job)
            return "running"

    async def _loaded(*_: str) -> None:
        return None

    told: list[dict[str, str]] = []

    async def _start_job() -> None:
        told.append(wrapper_env(tmp_path))

    monkeypatch.setattr(submission, "_LANES", _Lanes())
    monkeypatch.setattr(submission, "cuda_context_broken", lambda: False)
    monkeypatch.setattr(submission, "load_training_stack", _loaded)
    own, shared = tmp_path / "ana" / "datasets", tmp_path / "shared"
    ana = Profile(slug="ana", name="Ana", models_dir=tmp_path / "ana" / "models")
    scope = DatasetScope(readable=(own, shared), owner="ana", own=own)
    with dataset_scope(scope):
        submission._submit_job(
            "r1", "l", "classification", "simple", _start_job, profile=ana
        )
    submission._submit_job("r2", "l", "classification", "simple", _start_job)

    async def _start_all() -> None:
        for job in queued:
            await job.start()

    asyncio.run(_start_all())
    assert told[0][WRITE_ENV] == str(tmp_path / "ana" / "models")
    assert told[0][READ_ENV] == os.pathsep.join([str(own), str(shared)])
    assert told[1][WRITE_ENV] == told[1][READ_ENV] == ""


def test_the_cli_refuses_a_server_whose_wrapper_is_not_there(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from visionforge.__main__ import main

    monkeypatch.setenv(WRAPPER_ENV, "vf-no-such-sandbox")
    started: list[Any] = []
    argv = ["visionforge", "gui", "--server", "--trusted-proxy", "10.0.0.5"]
    with (
        patch.object(sys, "argv", argv),
        patch("visionforge.gui.server.start_server", lambda **kw: started.append(kw)),
        pytest.raises(SystemExit) as stop,
    ):
        main()
    assert stop.value.code == 2
    assert "vf-no-such-sandbox" in capsys.readouterr().err
    assert not started


def test_the_local_tool_ignores_the_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    from visionforge.__main__ import main

    monkeypatch.setenv(WRAPPER_ENV, "vf-no-such-sandbox")
    started: list[Any] = []
    with (
        patch.object(sys, "argv", ["visionforge", "gui", "--no-browser"]),
        patch("visionforge.gui.server.start_server", lambda **kw: started.append(kw)),
    ):
        main()
    assert len(started) == 1


@pytest.fixture
def log_lines() -> Iterator[list[str]]:
    seen: list[str] = []
    sink = logger.add(lambda message: seen.append(str(message)), level="INFO")
    yield seen
    logger.remove(sink)


@pytest.fixture
def server_app() -> Iterator[Any]:
    from visionforge.gui.server import app

    names = ("host_policy", "bind_host", "server_mode")
    saved = {n: getattr(app.state, n) for n in names if hasattr(app.state, n)}
    yield app
    for name in names:
        if name in saved:
            setattr(app.state, name, saved[name])
        elif hasattr(app.state, name):
            delattr(app.state, name)


def _start() -> None:
    import visionforge.gui.server as server_mod

    mode = ServerMode.from_options(enabled=True, trusted_proxies=["10.0.0.5"])
    with patch("uvicorn.Server"), patch("webbrowser.open"):
        server_mod.start_server(host="127.0.0.1", port=8123, server_mode=mode)


def test_a_server_says_which_program_its_jobs_start_through(
    server_app: Any, log_lines: list[str], record: Path
) -> None:
    _start()
    assert any(WRAPPER_ENV in line and "starts through" in line for line in log_lines)


def test_a_server_without_a_wrapper_says_nothing_about_it(
    server_app: Any, log_lines: list[str]
) -> None:
    _start()
    assert not any(WRAPPER_ENV in line for line in log_lines)
