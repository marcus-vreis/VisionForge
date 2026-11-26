"""Server mode: a dataset sent to be deleted after the run that uses it (ADR-249).

Each test runs the real app behind the trusted proxy: the dataset arrives by
the upload routes (ADR-246), the run by ``POST /api/experiment/run`` with a
stand-in trainer that writes the run's ``run.json`` and ends when told.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import io
import json
import os
import textwrap
import threading
import time
import zipfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import visionforge
from visionforge.gui.api import (
    execution,
    routes,
    submission,
    training_routes,
    upload_cleanup,
    upload_routes,
)
from visionforge.gui.api.run_queue import QueuedJob
from visionforge.gui.server_mode import ServerMode, user_slug
from visionforge.utils.messages import tr

PROXY = "10.0.0.5"
ANA = {"X-Forwarded-User": "ana@lab.org"}
BIA = {"X-Forwarded-User": "bia@lab.org"}
WAIT_S = 10.0


class _Trainer:
    """Stands in for the classification executor.

    Each run writes a ``run.json`` under the config's models folder, then holds
    until its name is released; a name in ``failing`` ends as a failed run
    whose folder the job does not name, as a crash mid-training leaves it.
    """

    def __init__(self) -> None:
        self.started: list[str] = []
        self.released: set[str] = set()
        self.failing: set[str] = set()
        self.lock = threading.Lock()

    async def __call__(self, config: Any, run_id: str, *_: Any, **__: Any) -> None:
        self.started.append(config.name)
        state = execution._job_state(run_id)
        state.run["status"] = "running"
        while True:
            with self.lock:
                if config.name in self.released:
                    break
            await asyncio.sleep(0.005)
        run_dir = Path(config.output.models_dir) / config.name / run_id
        run_dir.mkdir(parents=True)
        failed = config.name in self.failing
        record = {
            "id": run_id,
            "experiment": config.name,
            "timestamp": "2026-11-25T10:00:00",
            "status": "failed" if failed else "completed",
            "run_dir": str(run_dir.resolve()),
            "config": config.model_dump(mode="json"),
            "metrics": {},
            "history": [],
            "artifacts": {},
        }
        (run_dir / "run.json").write_text(json.dumps(record), encoding="utf-8")
        state.run = {
            "run_id": run_id,
            "status": "failed" if failed else "completed",
            "error": "RuntimeError: boom" if failed else None,
            "report": None if failed else {},
            "run_dir": None if failed else str(run_dir),
        }

    def release(self, name: str) -> None:
        with self.lock:
            self.released.add(name)

    def wait_for(self, count: int) -> None:
        deadline = time.monotonic() + WAIT_S
        while len(self.started) < count:
            assert time.monotonic() < deadline, f"started only {self.started}"
            time.sleep(0.01)


@pytest.fixture
def trainer() -> _Trainer:
    return _Trainer()


@pytest.fixture
def client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, trainer: _Trainer
) -> Iterator[TestClient]:
    """The real app in server mode with the stand-in trainer in the thread."""
    monkeypatch.setattr(execution, "_execute_experiment", trainer)
    with _server(tmp_path, monkeypatch) as test_client:
        yield test_client
        with trainer.lock:
            trainer.released.update(trainer.started)


@contextlib.contextmanager
def _server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """The real app in server mode, one loop for every request, an empty queue."""
    from visionforge.gui.server import app

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(routes, "_PROFILES_DIR", tmp_path / "profiles")
    monkeypatch.setattr(routes, "_MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(upload_routes, "PART_SIZE", 1024)
    monkeypatch.setattr(execution, "_JOB_STATES", {})
    monkeypatch.setattr(execution, "_latest_run_id", None)
    monkeypatch.setattr(submission, "cuda_context_broken", lambda: False)
    monkeypatch.setattr(submission, "_TRAINING_STACK", ())
    monkeypatch.setattr(
        training_routes, "_check_classification_classes", lambda config: None
    )
    execution._LANES.reset()
    had = hasattr(app.state, "server_mode")
    saved = getattr(app.state, "server_mode", None)
    app.state.server_mode = ServerMode.from_options(
        enabled=True, trusted_proxies=[PROXY]
    )
    test_client = TestClient(app, base_url="http://localhost", client=(PROXY, 50000))
    try:
        with test_client:
            yield test_client
    finally:
        execution._LANES.reset()
        if had:
            app.state.server_mode = saved
        else:
            del app.state.server_mode


def _own(client: TestClient, tmp_path: Path, headers: dict[str, str]) -> Path:
    client.get("/api/profiles", headers=headers)  # makes the profile and its folder
    slug: str = user_slug(headers["X-Forwarded-User"])
    return (tmp_path / "profiles" / slug / "datasets").resolve()


def _zip(name: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(f"{name}/train/a/1.jpg", b"\1" * 1500)
        archive.writestr(f"{name}/train/b/2.jpg", b"\2" * 1500)
        archive.writestr(f"{name}/val/a/3.jpg", b"\3" * 700)
    return buffer.getvalue()


def _upload(
    client: TestClient, headers: dict[str, str], name: str, delete_after_run: bool
) -> dict[str, Any]:
    data = _zip(name)
    body = {
        "name": name,
        "filename": f"{name}.zip",
        "size": len(data),
        "delete_after_run": delete_after_run,
    }
    started = client.post("/api/uploads", json=body, headers=headers)
    assert started.status_code == 200, started.text
    upload = started.json()
    assert upload["delete_after_run"] is delete_after_run
    size = upload["part_size"]
    for part in range(upload["parts"]):
        sent = client.put(
            f"/api/uploads/{upload['id']}/parts/{part}",
            content=data[part * size : (part + 1) * size],
            headers=headers,
        )
        assert sent.status_code == 200, sent.text
    done = client.post(f"/api/uploads/{upload['id']}/complete", headers=headers)
    assert done.status_code == 200, done.text
    return dict(done.json())


def _submit(
    client: TestClient, headers: dict[str, str], name: str, base_dir: Path
) -> str:
    body = {
        "name": name,
        "model": {"name": "resnet18", "num_classes": 2, "pretrained": False},
        "data": {"base_dir": str(base_dir)},
    }
    response = client.post("/api/experiment/run", json=body, headers=headers)
    assert response.status_code == 200, response.text
    return str(response.json()["run_id"])


def _wait(done: Callable[[], bool], what: str, timeout: float = WAIT_S) -> None:
    deadline = time.monotonic() + timeout
    while not done():
        assert time.monotonic() < deadline, what
        time.sleep(0.01)


def _wait_ended(run_id: str, timeout: float = WAIT_S) -> None:
    def ended() -> bool:
        return all(job.run_id != run_id for job in execution._LANES.jobs())

    _wait(ended, f"{run_id} never ended", timeout)


def _run_json(tmp_path: Path, headers: dict[str, str], name: str) -> dict[str, Any]:
    slug = user_slug(headers["X-Forwarded-User"])
    found = list((tmp_path / "profiles" / slug / "models" / name).rglob("run.json"))
    assert len(found) == 1, found
    return dict(json.loads(found[0].read_text(encoding="utf-8")))


def _listed(client: TestClient, headers: dict[str, str]) -> dict[str, bool]:
    listed = client.get("/api/datasets/own", headers=headers)
    assert listed.status_code == 200, listed.text
    return {d["name"]: d["delete_after_run"] for d in listed.json()["datasets"]}


def test_the_dataset_goes_after_a_completed_run_and_the_run_records_it(
    client: TestClient, tmp_path: Path, trainer: _Trainer
) -> None:
    own = _own(client, tmp_path, ANA)
    sent = _upload(client, ANA, "beans", delete_after_run=True)
    assert sent["delete_after_run"] is True
    size = sent["size_bytes"]
    # Bia's dataset of the same name, sent to go too, is not touched by Ana's run.
    bia_own = _own(client, tmp_path, BIA)
    _upload(client, BIA, "beans", delete_after_run=True)
    assert _listed(client, ANA) == {"beans": True}
    # The mark is beside the dataset folder, not inside it: no loader reads it.
    assert sorted(p.name for p in (own / "beans").iterdir()) == ["train", "val"]

    run_id = _submit(client, ANA, "exp", own / "beans")
    trainer.wait_for(1)
    trainer.release("exp")
    _wait_ended(run_id)

    assert not (own / "beans").exists()
    assert list(own.iterdir()) == []
    assert not upload_cleanup.is_marked(own, "beans")
    assert _listed(client, ANA) == {}
    assert (bia_own / "beans").is_dir()
    assert _listed(client, BIA) == {"beans": True}

    record = _run_json(tmp_path, ANA, "exp")["dataset_cleanup"]
    assert len(record) == 1
    entry = record[0]
    assert entry["dataset"] == "beans"
    assert entry["path"] == "datasets/beans"
    assert entry["size_bytes"] == size
    assert (entry["status"], entry["error"]) == ("deleted", None)
    assert entry["at"]

    status = client.get(f"/api/experiment/status?run_id={run_id}", headers=ANA)
    assert status.json()["status"] == "completed"
    assert status.json()["dataset_cleanup"] == record
    result = client.get(f"/api/experiment/result/{run_id}", headers=ANA)
    assert result.status_code == 200, result.text
    assert result.json()["dataset_cleanup"] == record
    detail = client.get(f"/api/runs/{run_id}", headers=ANA)
    assert detail.status_code == 200, detail.text
    assert detail.json()["dataset_cleanup"] == record


def test_the_dataset_goes_after_a_failed_run_too(
    client: TestClient, tmp_path: Path, trainer: _Trainer
) -> None:
    own = _own(client, tmp_path, ANA)
    _upload(client, ANA, "beans", delete_after_run=True)
    trainer.failing.add("exp")

    run_id = _submit(client, ANA, "exp", own / "beans")
    trainer.wait_for(1)
    trainer.release("exp")
    _wait_ended(run_id)

    assert not (own / "beans").exists()
    status = client.get(f"/api/experiment/status?run_id={run_id}", headers=ANA).json()
    assert status["status"] == "failed"
    assert [e["status"] for e in status["dataset_cleanup"]] == ["deleted"]
    # The job names no run folder: the run.json it wrote is found and records it.
    record = _run_json(tmp_path, ANA, "exp")
    assert record["status"] == "failed"
    assert record["dataset_cleanup"] == status["dataset_cleanup"]


def test_the_dataset_stays_while_another_job_of_theirs_uses_it(
    client: TestClient, tmp_path: Path, trainer: _Trainer
) -> None:
    own = _own(client, tmp_path, ANA)
    _upload(client, ANA, "beans", delete_after_run=True)
    first = _submit(client, ANA, "first", own / "beans")
    trainer.wait_for(1)
    second = _submit(client, ANA, "second", own / "beans")

    trainer.release("first")
    _wait_ended(first)
    assert (own / "beans").is_dir()
    assert upload_cleanup.is_marked(own, "beans")
    assert "dataset_cleanup" not in _run_json(tmp_path, ANA, "first")

    trainer.wait_for(2)
    trainer.release("second")
    _wait_ended(second)
    assert not (own / "beans").exists()
    entries = _run_json(tmp_path, ANA, "second")["dataset_cleanup"]
    assert [e["status"] for e in entries] == ["deleted"]


def test_a_dataset_sent_without_the_choice_stays(
    client: TestClient, tmp_path: Path, trainer: _Trainer
) -> None:
    own = _own(client, tmp_path, ANA)
    sent = _upload(client, ANA, "beans", delete_after_run=False)
    assert sent["delete_after_run"] is False
    assert _listed(client, ANA) == {"beans": False}

    run_id = _submit(client, ANA, "exp", own / "beans")
    trainer.wait_for(1)
    trainer.release("exp")
    _wait_ended(run_id)

    assert (own / "beans").is_dir()
    assert "dataset_cleanup" not in _run_json(tmp_path, ANA, "exp")
    status = client.get(f"/api/experiment/status?run_id={run_id}", headers=ANA)
    assert status.json()["dataset_cleanup"] == []


def test_a_deletion_that_fails_is_recorded_and_the_run_still_completes(
    client: TestClient,
    tmp_path: Path,
    trainer: _Trainer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    own = _own(client, tmp_path, ANA)
    _upload(client, ANA, "beans", delete_after_run=True)

    def refuse(source: Path, target: Path) -> None:
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr(upload_cleanup.os, "rename", refuse)
    run_id = _submit(client, ANA, "exp", own / "beans")
    trainer.wait_for(1)
    trainer.release("exp")
    _wait_ended(run_id)

    status = client.get(f"/api/experiment/status?run_id={run_id}", headers=ANA).json()
    assert status["status"] == "completed"
    [entry] = status["dataset_cleanup"]
    assert (entry["status"], entry["error"]) == ("failed", "Access is denied")
    assert _run_json(tmp_path, ANA, "exp")["dataset_cleanup"] == [entry]
    # Intact and still marked: the next run that uses it tries again.
    assert (own / "beans" / "train").is_dir()
    assert _listed(client, ANA) == {"beans": True}


def test_deleting_a_marked_dataset_by_hand_clears_its_mark(
    client: TestClient, tmp_path: Path
) -> None:
    own = _own(client, tmp_path, ANA)
    _upload(client, ANA, "beans", delete_after_run=True)
    assert client.delete("/api/datasets/own/beans", headers=ANA).status_code == 200
    assert not upload_cleanup.is_marked(own, "beans")
    # A new dataset of that name sent to stay is not deleted by an old mark.
    _upload(client, ANA, "beans", delete_after_run=False)
    assert _listed(client, ANA) == {"beans": False}


# ── the training in a child process (ADR-248) ───────────────────────────────

CHILD_MODULE = "vf_fake_cleanup_block"
CHILD_WAIT_S = 120.0

_CHILD_SOURCE = textwrap.dedent(
    '''
    """A classification block for the cleanup tests, run by the job's child."""
    import json
    import os
    from pathlib import Path
    from types import SimpleNamespace


    class FakeBlock:
        def setup(self, config):
            self.config = config
            self._train_result = None

        def run(self):
            c = self.config
            run_dir = Path(c.output.models_dir) / c.name / "child_run"
            run_dir.mkdir(parents=True)
            seen = sorted(
                p.relative_to(c.data.base_dir).as_posix()
                for p in Path(c.data.base_dir).rglob("*.jpg")
            )
            record = {"id": "child_run", "experiment": c.name, "seen": seen,
                      "pid": os.getpid()}
            (run_dir / "run.json").write_text(json.dumps(record), encoding="utf-8")
            if c.name == "crash":
                os._exit(3)  # dies mid-run: no end message, no run folder named
            self._train_result = SimpleNamespace(model_path=run_dir / "best.pt")

        def report(self):
            return {}
    '''
)


@pytest.fixture
def child_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    """The real app in server mode whose jobs train in a real child Python."""
    blocks = tmp_path / "blocks"
    blocks.mkdir()
    (blocks / f"{CHILD_MODULE}.py").write_text(_CHILD_SOURCE, encoding="utf-8")
    src = Path(visionforge.__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(blocks))
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(src), str(blocks)]))
    fake = importlib.import_module(CHILD_MODULE)
    monkeypatch.setattr(
        execution, "_block_types", lambda: {"classification": fake.FakeBlock}
    )
    with _server(tmp_path, monkeypatch) as test_client:
        yield test_client


@pytest.mark.parametrize(
    ("name", "status"), [("exp", "completed"), ("crash", "failed")]
)
def test_the_dataset_goes_after_a_run_in_the_child_process(
    child_client: TestClient, tmp_path: Path, name: str, status: str
) -> None:
    own = _own(child_client, tmp_path, ANA)
    _upload(child_client, ANA, "beans", delete_after_run=True)

    run_id = _submit(child_client, ANA, name, own / "beans")
    _wait_ended(run_id, CHILD_WAIT_S)

    record = _run_json(tmp_path, ANA, name)
    # The child read the dataset, in a process other than this one.
    assert record["seen"] == ["train/a/1.jpg", "train/b/2.jpg", "val/a/3.jpg"]
    assert record["pid"] != os.getpid()
    assert not (own / "beans").exists()
    assert not upload_cleanup.is_marked(own, "beans")
    [entry] = record["dataset_cleanup"]
    assert (entry["dataset"], entry["status"], entry["error"]) == (
        "beans",
        "deleted",
        None,
    )
    run = child_client.get(f"/api/experiment/status?run_id={run_id}", headers=ANA)
    assert run.json()["status"] == status
    assert run.json()["dataset_cleanup"] == [entry]
    if status == "failed":
        assert run.json()["error"] == tr("job.child_died", code=3)


# ── the run folder a job names ───────────────────────────────────────────────


async def _nothing() -> None:
    """A job's coroutine that does nothing."""


def test_a_run_folder_outside_the_models_folder_is_not_recorded_in(
    tmp_path: Path,
) -> None:
    models = tmp_path / "models"
    inside = models / "exp" / "run_1"
    outside = tmp_path / "elsewhere"
    for folder in (inside, outside):
        folder.mkdir(parents=True)
        (folder / "run.json").write_text("{}", encoding="utf-8")
    job = QueuedJob(
        run_id="r1",
        label="exp",
        task="classification",
        strategy="simple",
        start=_nothing,
        models_dir=models,
    )

    assert upload_cleanup._run_json(job, str(inside)) == inside / "run.json"
    # A child process names its own run folder: one outside is not believed.
    assert upload_cleanup._run_json(job, str(outside)) is None
    assert upload_cleanup._run_json(job, str(models / ".." / "elsewhere")) is None
