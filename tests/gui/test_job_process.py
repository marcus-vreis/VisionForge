"""A server job's training in a child process (ADR-248).

These start real child Pythons (``python -m visionforge.gui.api.job_worker``)
running a fake block written to a temporary module: CPU only, a second or two
each. The fake reads its behaviour from its config: train a few epochs, raise,
die without a word, ignore a stop, print to stdout, or report what it can see.
"""

from __future__ import annotations

import asyncio
import importlib
import os
import sys
import textwrap
import threading
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import visionforge
from visionforge.core.cancellation import CancellationToken
from visionforge.gui.api import execution, job_process, submission, training_routes
from visionforge.gui.api.dataset_scope import DatasetScope, dataset_scope
from visionforge.gui.api.job_process import (
    ChildJob,
    ChildOutcome,
    child_env,
    child_spec,
    pinned_config,
)
from visionforge.gui.api.run_queue import QueuedJob
from visionforge.utils.config import ExperimentConfig
from visionforge.utils.messages import tr

FAKE_MODULE = "vf_fake_job_block"
SRC = Path(visionforge.__file__).resolve().parents[1]
SECRETS = ("HF_TOKEN", "GITHUB_TOKEN", "KAGGLE_API_TOKEN", "WANDB_API_KEY")

_FAKE_SOURCE = textwrap.dedent(
    '''
    """A block that stands in for a trainer in the child-process tests."""
    import os
    import time
    from pathlib import Path
    from types import SimpleNamespace

    from pydantic import BaseModel


    class FakeModel(BaseModel):
        name: str = "fake-net"


    class FakeConfig(BaseModel):
        name: str = "fake"
        block: str = "classification"
        mode: str = "good"
        epochs: int = 3
        pause: float = 0.0
        out: Path = Path("out")
        model: FakeModel = FakeModel()


    class FakeBlock:
        def setup(self, config):
            self.config = config
            self._progress_callback = None
            self._cancel_token = None
            self._resume_dir = None
            self._train_result = None
            self._report = {}

        def _emit(self, event):
            if self._progress_callback is not None:
                self._progress_callback(event)

        def run(self):
            c = self.config
            if c.mode == "raise":
                raise ValueError("bad batch")
            if c.mode == "crash":
                os._exit(3)
            if c.mode == "env":
                self._report = {
                    "seen": {k: k in os.environ for k in (*SECRETS, "PATH")},
                    "home": os.environ.get("VISIONFORGE_HOME"),
                    "cwd": os.getcwd(),
                    "out": str(c.out),
                    "resume": str(self._resume_dir),
                }
                return
            if c.mode == "numpy":
                # A C extension whose DLLs load while the job is running, as
                # torch's do: numpy's OpenBLAS starts a CRT of its own.
                import numpy

                self._report = {"sum": int(numpy.arange(4).sum())}
                return
            if c.mode == "noisy":
                print("an epoch, said the library")
                os.write(1, b"raw bytes on fd 1\\n")
            c.out.mkdir(parents=True, exist_ok=True)
            epoch, stopped = 0, False
            while epoch < c.epochs or c.mode == "stubborn":
                if c.mode != "stubborn" and self._cancel_token.cancelled:
                    stopped = True
                    break
                epoch += 1
                self._emit({"type": "epoch_end", "epoch": epoch, "loss": 1 / epoch})
                time.sleep(c.pause)
            self._train_result = SimpleNamespace(model_path=c.out / "best.pt")
            self._report = {
                "train": {"run_dir": str(c.out), "stopped": stopped, "epochs": epoch}
            }

        def report(self):
            return self._report


    SECRETS = ("HF_TOKEN", "GITHUB_TOKEN", "KAGGLE_API_TOKEN", "WANDB_API_KEY")
    '''
)


@pytest.fixture(scope="module")
def fake_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    folder = Path(tmp_path_factory.mktemp("fake_block"))
    (folder / f"{FAKE_MODULE}.py").write_text(_FAKE_SOURCE, encoding="utf-8")
    return folder


@pytest.fixture
def fake(
    fake_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[ModuleType]:
    """The fake module, importable here and in the child; tokens in the server's env."""
    monkeypatch.syspath_prepend(str(fake_dir))
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(SRC), str(fake_dir)]))
    for name in SECRETS:
        monkeypatch.setenv(name, f"secret-{name.lower()}")
    monkeypatch.chdir(tmp_path)
    yield importlib.import_module(FAKE_MODULE)


def _spec(fake: ModuleType, **config: Any) -> dict[str, Any]:
    return child_spec(
        fake.FakeBlock, fake.FakeConfig(**config), None, report_section="train"
    )


def _run(
    spec: dict[str, Any], token: CancellationToken | None = None, **kwargs: Any
) -> tuple[ChildOutcome, list[dict[str, Any]], ChildJob]:
    events: list[dict[str, Any]] = []
    job = ChildJob(spec, events.append, token or CancellationToken(), **kwargs)
    return job.supervise(), events, job


# ── the environment and the config ──────────────────────────────────────────


def test_the_child_environment_keeps_what_training_needs_and_no_secret(
    tmp_path: Path,
) -> None:
    environ = {
        "PATH": "/usr/bin",
        "SystemRoot": r"C:\Windows",
        "PYTHONPATH": "/src",
        "CUDA_VISIBLE_DEVICES": "0",
        "HF_HOME": "/cache/hf",
        "VISIONFORGE_LANG": "en",
        "HF_TOKEN": "hf_x",
        "HUGGING_FACE_HUB_TOKEN": "hf_y",
        "GITHUB_TOKEN": "ghp_x",
        "KAGGLE_API_TOKEN": "k",
        "WANDB_API_KEY": "w",
        "AWS_SECRET_ACCESS_KEY": "a",
        "PYTHON_KEYRING_PASSWORD": "p",
        "HTTPS_PROXY": "http://user:pw@proxy:3128",
        "HTTP_PROXY": "http://proxy:3128",
        "VISIONFORGE_HOME": "/home/me/.visionforge",
        "SOMETHING_ELSE": "1",
    }
    env = child_env(environ, tmp_path)
    assert env == {
        "PATH": "/usr/bin",
        "SystemRoot": r"C:\Windows",
        "PYTHONPATH": "/src",
        "CUDA_VISIBLE_DEVICES": "0",
        "HF_HOME": "/cache/hf",
        "VISIONFORGE_LANG": "en",
        "HTTP_PROXY": "http://proxy:3128",
        "VISIONFORGE_HOME": str(tmp_path),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1",
    }


def test_the_config_crosses_with_its_paths_made_absolute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "beans").mkdir(parents=True)
    config = ExperimentConfig.model_validate(
        {
            "name": "x",
            "model": {"name": "resnet18", "num_classes": 2, "pretrained": False},
            "data": {"base_dir": "data/beans"},
            "cross_validation": {"group_manifest": "groups.csv"},
        }
    )
    data = pinned_config(config)
    assert data["data"]["base_dir"] == str(tmp_path / "data" / "beans")
    assert data["output"]["models_dir"] == str(tmp_path / "outputs" / "models")
    # Resolved against the dataset folder by its loader, so it stays relative.
    assert data["cross_validation"]["group_manifest"] == "groups.csv"
    assert ExperimentConfig.model_validate(data).model_dump(mode="json") == data


# ── a real child ────────────────────────────────────────────────────────────


def test_events_arrive_in_order_and_the_result_comes_back(fake: ModuleType) -> None:
    outcome, events, _ = _run(_spec(fake, epochs=3))
    assert outcome.status == "completed", outcome.error
    assert [e["epoch"] for e in events] == [1, 2, 3]
    assert events[2] == {"type": "epoch_end", "epoch": 3, "loss": pytest.approx(1 / 3)}
    out = str(Path.cwd() / "out")
    assert outcome.run_dir == out
    assert outcome.report == {"train": {"run_dir": out, "stopped": False, "epochs": 3}}
    assert outcome.returncode == 0


def test_a_library_printing_to_stdout_does_not_break_the_events(
    fake: ModuleType,
) -> None:
    outcome, events, _ = _run(_spec(fake, mode="noisy", epochs=2))
    assert outcome.status == "completed", outcome.error
    assert [e["epoch"] for e in events] == [1, 2]


def test_a_block_that_loads_a_native_library_mid_run_does_not_hang(
    fake: ModuleType,
) -> None:
    # On Windows, a thread of the child blocked reading its stdin pipe froze the
    # load of numpy's OpenBLAS DLL (its CRT start-up queries the standard
    # handles): a real training never imported torch and stayed "running".
    result: list[ChildOutcome] = []
    job = ChildJob(_spec(fake, mode="numpy"), lambda e: None, CancellationToken())
    runner = threading.Thread(target=lambda: result.append(job.supervise()))
    runner.start()
    runner.join(60)
    if runner.is_alive():
        job.kill()
        runner.join(30)
    assert result, "the child hung loading numpy"
    assert result[0].status == "completed", result[0].error
    assert result[0].report == {"sum": 6}


def test_a_stop_reaches_the_block_at_its_stop_point(fake: ModuleType) -> None:
    token = CancellationToken()
    events: list[dict[str, Any]] = []

    def put(event: dict[str, Any]) -> None:
        events.append(event)
        token.cancel()

    job = ChildJob(_spec(fake, epochs=200, pause=0.05), put, token)
    outcome = job.supervise()
    assert outcome.status == "completed", outcome.error
    assert outcome.report is not None
    assert outcome.report["train"]["stopped"] is True
    assert 1 <= len(events) < 200
    assert not outcome.killed


def test_a_child_that_ignores_the_stop_is_killed_after_the_grace(
    fake: ModuleType,
) -> None:
    token = CancellationToken()
    first = threading.Event()

    def put(event: dict[str, Any]) -> None:
        first.set()
        token.cancel()

    job = ChildJob(_spec(fake, mode="stubborn", pause=0.05), put, token, grace=1.0)
    outcome = job.supervise()
    assert first.is_set()
    assert outcome.killed is True
    assert outcome.status == "failed"
    assert outcome.error == tr("job.child_killed", seconds=1)
    assert job._proc is not None
    assert job._proc.poll() is not None


def test_a_server_that_drops_the_job_kills_its_child(
    fake: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    started: list[ChildJob] = []

    class _Spy(ChildJob):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            started.append(self)

    monkeypatch.setattr(job_process, "ChildJob", _Spy)

    async def scenario() -> None:
        first = asyncio.Event()
        loop = asyncio.get_running_loop()
        spec = _spec(fake, mode="stubborn", pause=0.05)

        def put(event: dict[str, Any]) -> None:
            loop.call_soon_threadsafe(first.set)

        task = asyncio.create_task(
            job_process.run_in_child(spec, put, CancellationToken())
        )
        await asyncio.wait_for(first.wait(), 30)
        task.cancel()  # what a queue reset or a server shutdown does
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    proc = started[0]._proc
    assert proc is not None
    assert proc.wait(timeout=10) is not None
    assert started[0]._killed is True


def test_a_crashing_child_marks_the_run_failed_and_the_server_goes_on(
    fake: ModuleType,
) -> None:
    outcome, events, _ = _run(_spec(fake, mode="crash"))
    assert (outcome.status, outcome.returncode, events) == ("failed", 3, [])
    assert outcome.error == tr("job.child_died", code=3)
    # This process is the server: it is still here and can start the next job.
    again, _, _ = _run(_spec(fake, epochs=1))
    assert again.status == "completed", again.error


def test_a_block_that_raises_reports_its_error_as_the_thread_did(
    fake: ModuleType,
) -> None:
    outcome, _, _ = _run(_spec(fake, mode="raise"))
    assert outcome.status == "failed"
    assert outcome.error == "ValueError: bad batch"


def test_the_child_has_no_token_and_a_folder_of_its_own(fake: ModuleType) -> None:
    assert os.environ["HF_TOKEN"] == "secret-hf_token"
    spec = child_spec(
        fake.FakeBlock,
        fake.FakeConfig(mode="env"),
        Path("runs/old"),
        report_section="train",
    )
    outcome, _, _ = _run(spec)
    assert outcome.status == "completed", outcome.error
    report = outcome.report or {}
    assert report["seen"] == dict.fromkeys(SECRETS, False) | {"PATH": True}
    cwd = Path(report["cwd"])
    assert cwd != Path.cwd()
    assert Path(report["home"]) == cwd / "home"
    assert not cwd.exists(), "the working folder is removed when the job ends"
    # Relative paths mean what they meant in the server's folder.
    assert report["out"] == str(Path.cwd() / "out")
    assert report["resume"] == str(Path.cwd() / "runs" / "old")


# ── through the executors ───────────────────────────────────────────────────


@pytest.fixture
def jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(execution, "_JOB_STATES", {})
    monkeypatch.setattr(execution, "_latest_run_id", None)


def _start(run_id: str, *, isolated: bool) -> execution.JobState:
    async def _nothing() -> None:
        return None

    job = QueuedJob(
        run_id=run_id,
        label=run_id,
        task="classification",
        strategy="simple",
        start=_nothing,
        isolated=isolated,
    )
    execution._begin_job(job)
    return execution._job_state(run_id)


async def _events(state: execution.JobState) -> list[dict[str, Any]]:
    return [event async for event in state.log.read()]


@pytest.mark.usefixtures("jobs")
def test_a_server_task_run_trains_in_a_child_and_streams_its_epochs(
    fake: ModuleType,
) -> None:
    state = _start("seg_1", isolated=True)

    async def scenario() -> list[dict[str, Any]]:
        await training_routes._execute_block(
            fake.FakeBlock, "segmentation", fake.FakeConfig(epochs=2), "seg_1", None
        )
        return await _events(state)

    events = asyncio.run(scenario())
    assert [e["epoch"] for e in events] == [1, 2]
    assert state.run["status"] == "completed"
    assert state.run["run_dir"] == str(Path.cwd() / "out")
    assert "vf_fake_job_block" not in str(state.run)


@pytest.mark.usefixtures("jobs")
def test_a_crashed_child_fails_the_task_run_and_closes_its_stream(
    fake: ModuleType,
) -> None:
    state = _start("seg_2", isolated=True)

    async def scenario() -> list[dict[str, Any]]:
        await training_routes._execute_block(
            fake.FakeBlock, "segmentation", fake.FakeConfig(mode="crash"), "seg_2", None
        )
        return await _events(state)

    assert asyncio.run(scenario()) == []
    assert state.run["status"] == "failed"
    assert state.run["error"] == tr("job.child_died", code=3)


@pytest.mark.usefixtures("jobs")
def test_a_server_classification_trains_in_a_child_and_exports_at_the_end(
    fake: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        execution, "_block_types", lambda: {"classification": fake.FakeBlock}
    )
    exported: list[tuple[Any, Path]] = []
    monkeypatch.setattr(
        execution,
        "export_run_record",
        lambda trackers, run_dir: exported.append((trackers, run_dir)),
    )
    state = _start("cls_1", isolated=True)
    trackers = (object(),)

    async def scenario() -> list[dict[str, Any]]:
        await execution._execute_experiment(
            fake.FakeConfig(epochs=2),  # type: ignore[arg-type]
            "cls_1",
            trackers=trackers,  # type: ignore[arg-type]
        )
        return await _events(state)

    events = asyncio.run(scenario())
    assert [e["epoch"] for e in events] == [1, 2]
    out = Path.cwd() / "out"
    # Classification reads its folder from the training result: the checkpoint's.
    assert (state.run["status"], state.run["run_dir"]) == ("completed", str(out))
    assert exported == [(trackers, out)]


@pytest.mark.slow
@pytest.mark.usefixtures("jobs")
def test_a_real_classification_trains_in_a_child_on_the_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real launch: the training stack (torch, numpy's OpenBLAS) loads in the child."""
    from visionforge.gui.api.sample_dataset import create_sample_dataset

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join([str(SRC), os.environ.get("PYTHONPATH", "")])
    )
    sample = create_sample_dataset(tmp_path / "datasets", "classification")
    config = ExperimentConfig.model_validate(
        {
            "name": "child_real",
            "model": {
                "name": "resnet18",
                "num_classes": len(sample.classes) or 2,
                "pretrained": False,
            },
            "training": {"epochs": 2, "batch_size": 4},
            "data": {
                "base_dir": sample.path,
                "num_workers": 0,
                "pin_memory": False,
                "transforms": {"image_size": 32},
            },
            "device": {"kind": "cpu"},
        }
    )
    state = _start("real_1", isolated=True)

    async def scenario() -> list[dict[str, Any]]:
        await asyncio.wait_for(execution._execute_experiment(config, "real_1"), 300)
        return await _events(state)

    events = asyncio.run(scenario())
    assert state.run["status"] == "completed", state.run["error"]
    epochs = [e["epoch"] for e in events if e.get("event") == "epoch_end"]
    assert epochs == [1, 2]
    run_dir = Path(state.run["run_dir"])
    assert (run_dir / "run.json").is_file()
    assert run_dir.is_relative_to(tmp_path)


@pytest.mark.usefixtures("jobs")
def test_a_strategy_and_a_local_job_keep_the_training_thread(
    fake: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        execution,
        "_block_types",
        lambda: {"classification": fake.FakeBlock, "grid_search": fake.FakeBlock},
    )

    async def no_child(*_: Any, **__: Any) -> None:
        raise AssertionError("this job must not start a child")

    monkeypatch.setattr(execution, "_execute_in_child", no_child)
    for run_id, isolated, block in (
        ("grid_1", True, "grid_search"),
        ("local_1", False, "classification"),
    ):
        state = _start(run_id, isolated=isolated)
        config = fake.FakeConfig(block=block, epochs=1)
        asyncio.run(execution._execute_experiment(config, run_id))  # type: ignore[arg-type]
        assert state.run["status"] == "completed", state.run["error"]
        # The config object itself reached the block: its path was not pinned.
        assert state.run["report"]["train"]["run_dir"] == "out"


def test_only_a_server_local_job_is_marked_for_a_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queued: list[QueuedJob] = []

    class _Lanes:
        def count_of(self, profile: str) -> int:
            return 0

        def submit(self, job: QueuedJob, lane: str, fair: bool = False) -> str:
            queued.append(job)
            return "running"

    async def _start_job() -> None:
        return None

    monkeypatch.setattr(submission, "_LANES", _Lanes())
    monkeypatch.setattr(submission, "cuda_context_broken", lambda: False)
    submission._submit_job("local", "l", "classification", "simple", _start_job)
    server = DatasetScope(readable=(), owner="ana", max_queued=3)
    with dataset_scope(server):
        submission._submit_job("server", "s", "classification", "simple", _start_job)
        submission._submit_job(
            "colab", "c", "classification", "colab", _start_job, lane="colab:ana"
        )
    assert [(job.run_id, job.isolated) for job in queued] == [
        ("local", False),
        ("server", True),
        ("colab", False),
    ]


def test_the_worker_module_runs_nothing_when_imported() -> None:
    # Windows' spawn imports it again as __mp_main__ in each DataLoader worker.
    before = sys.stdout
    module = importlib.import_module(job_process.WORKER_MODULE)
    assert sys.stdout is before
    assert callable(module.main)
