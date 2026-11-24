"""Server mode: what a finished run offers reaches only its owner's files (ADR-218).

Testing, Grad-CAM, batch prediction and ONNX export load the run's own
checkpoint, read images from a folder in the scope and write into the run's
folder. Resuming reads the stored config again like a new one.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel, ValidationError

from visionforge.gui.api import evaluation_routes, routes, submission
from visionforge.gui.api.dataset_scope import DatasetScope, dataset_scope
from visionforge.gui.api.schemas import (
    BatchPredictRequest,
    ExportOnnxRequest,
    GradCamRequest,
    RunResponse,
    RunTestRequest,
)
from visionforge.gui.server_mode import ServerMode, user_slug

PROXY = "10.0.0.5"
ANA = {"X-Forwarded-User": "ana@lab.org"}
BIA = {"X-Forwarded-User": "bia@lab.org"}
RUN_ID = "20261119_120000_000001"

# Each action, the executor it reaches and a request that names a folder.
ACTIONS = {
    "test": ("_execute_run_test", "data_dir"),
    "gradcam": ("_execute_run_gradcam", "input_dir"),
    "batch_predict": ("_execute_batch_predict", "input_dir"),
    "export_onnx": ("_execute_onnx_export", None),
}


@pytest.fixture
def world(tmp_path: Path) -> dict[str, Path]:
    shared = tmp_path / "shared"
    (shared / "beans" / "test" / "healthy").mkdir(parents=True)
    secret = tmp_path / "secret"
    (secret / "images").mkdir(parents=True)
    (secret / "stolen.pt").write_bytes(b"not a checkpoint")
    return {"shared": shared.resolve(), "secret": secret.resolve(), "root": tmp_path}


@pytest.fixture
def served(
    world: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[TestClient, list[str]]]:
    """The real app in server mode; the executors are recorded, never run."""
    from visionforge.gui.server import app

    reached: list[str] = []

    def recorder(name: str) -> Any:
        def execute(run_dir: Path, req: BaseModel) -> Any:
            reached.append(name)
            raise ValueError(f"reached {name}")  # answers 400: no model is built

        return execute

    for action, (executor, _) in ACTIONS.items():
        monkeypatch.setattr(evaluation_routes, executor, recorder(action))
    monkeypatch.setattr(routes, "_PROFILES_DIR", tmp_path / "profiles")
    monkeypatch.setattr(routes, "_MODELS_DIR", tmp_path / "models")
    had = hasattr(app.state, "server_mode")
    saved = getattr(app.state, "server_mode", None)
    app.state.server_mode = ServerMode.from_options(
        enabled=True, trusted_proxies=[PROXY], datasets_dirs=[world["shared"]]
    )
    client = TestClient(app, base_url="http://localhost", client=(PROXY, 50000))
    client.get("/api/session", headers=ANA)  # creates ana's profile folder
    yield client, reached
    if had:
        app.state.server_mode = saved
    else:
        del app.state.server_mode


def _run(world: dict[str, Path], checkpoint: Path | None = None, **record: Any) -> Path:
    """A finished run in ana's folder; its checkpoint is its own unless given."""
    models = world["root"] / "profiles" / user_slug("ana@lab.org") / "models"
    run_dir = models / "beans" / RUN_ID
    run_dir.mkdir(parents=True)
    own = run_dir / "best_model.pth"
    own.write_bytes(b"weights")
    data = {
        "run_id": RUN_ID,
        "status": "completed",
        "config": {"name": "beans", "task": "multiclass"},
        "artifacts": {"model": str(checkpoint or own)},
        **record,
    }
    (run_dir / "run.json").write_text(json.dumps(data), encoding="utf-8")
    return run_dir


def _body(action: str, folder: Path) -> dict[str, Any]:
    field = ACTIONS[action][1]
    return {field: str(folder)} if field else {}


@pytest.mark.parametrize("request_type", [RunTestRequest, GradCamRequest])
def test_the_folder_an_action_reads_must_be_in_the_scope(
    world: dict[str, Path], request_type: type[BaseModel]
) -> None:
    field = "data_dir" if request_type is RunTestRequest else "input_dir"
    inside, outside = world["shared"] / "beans" / "test", world["secret"] / "images"
    with dataset_scope(DatasetScope(readable=(world["shared"],))):
        assert request_type(**{field: str(inside)})
        with pytest.raises(ValidationError):
            request_type(**{field: str(outside)})
        with pytest.raises(ValidationError):
            request_type(**{field: "beans/test"})
    assert request_type(**{field: str(outside)})  # the local tool reads anywhere


def test_on_a_server_an_action_cannot_choose_where_its_file_goes(
    world: dict[str, Path],
) -> None:
    inside = str(world["shared"] / "beans" / "test")
    target = str(world["secret"] / "out.bin")
    with dataset_scope(DatasetScope(readable=(world["shared"],))):
        assert ExportOnnxRequest().output_onnx is None
        assert BatchPredictRequest(input_dir=inside, output_csv=" ").output_csv == " "
        with pytest.raises(ValidationError):
            ExportOnnxRequest(output_onnx=target)
        with pytest.raises(ValidationError):
            BatchPredictRequest(input_dir=inside, output_csv=target)
    assert ExportOnnxRequest(output_onnx=target).output_onnx == target
    assert BatchPredictRequest(input_dir=inside, output_csv=target).output_csv == target


@pytest.mark.parametrize("action", sorted(ACTIONS))
def test_an_action_on_the_persons_own_run_reaches_its_executor(
    served: tuple[TestClient, list[str]], world: dict[str, Path], action: str
) -> None:
    client, reached = served
    _run(world)
    response = client.post(
        f"/api/runs/{RUN_ID}/{action}",
        json=_body(action, world["shared"] / "beans" / "test"),
        headers=ANA,
    )
    assert (response.status_code, reached) == (400, [action]), response.text
    assert f"reached {action}" in response.json()["detail"]


@pytest.mark.parametrize("action", sorted(ACTIONS))
def test_nobody_else_can_act_on_the_run(
    served: tuple[TestClient, list[str]], world: dict[str, Path], action: str
) -> None:
    client, reached = served
    _run(world)
    response = client.post(
        f"/api/runs/{RUN_ID}/{action}",
        json=_body(action, world["shared"] / "beans" / "test"),
        headers=BIA,
    )
    assert (response.status_code, reached) == (404, [])


@pytest.mark.parametrize("action", ["test", "gradcam", "batch_predict"])
def test_an_action_cannot_read_a_folder_outside_the_scope(
    served: tuple[TestClient, list[str]], world: dict[str, Path], action: str
) -> None:
    client, reached = served
    _run(world)
    response = client.post(
        f"/api/runs/{RUN_ID}/{action}",
        json=_body(action, world["secret"] / "images"),
        headers=ANA,
    )
    assert (response.status_code, reached) == (422, [])


@pytest.mark.parametrize(
    ("action", "extra"),
    [("export_onnx", "output_onnx"), ("batch_predict", "output_csv")],
)
def test_an_action_cannot_write_outside_the_run_folder(
    served: tuple[TestClient, list[str]],
    world: dict[str, Path],
    action: str,
    extra: str,
) -> None:
    client, reached = served
    _run(world)
    body = _body(action, world["shared"] / "beans" / "test")
    body[extra] = str(world["shared"] / "planted.bin")
    response = client.post(f"/api/runs/{RUN_ID}/{action}", json=body, headers=ANA)
    assert (response.status_code, reached) == (422, [])


@pytest.mark.parametrize("action", sorted(ACTIONS))
def test_a_checkpoint_that_is_not_the_owners_is_never_loaded(
    served: tuple[TestClient, list[str]], world: dict[str, Path], action: str
) -> None:
    """Loading a detection checkpoint unpickles it: only a file in the person's
    own folder is loaded, whatever ``run.json`` says."""
    client, reached = served
    _run(world, checkpoint=world["secret"] / "stolen.pt")
    response = client.post(
        f"/api/runs/{RUN_ID}/{action}",
        json=_body(action, world["shared"] / "beans" / "test"),
        headers=ANA,
    )
    assert (response.status_code, reached) == (409, []), response.text
    assert RUN_ID in response.json()["detail"]


def _stopped_run(world: dict[str, Path], base_dir: Path) -> None:
    config = {
        "name": "beans",
        "model": {"name": "resnet18", "num_classes": 2, "pretrained": False},
        "data": {"base_dir": str(base_dir)},
    }
    run_dir = _run(world)
    data = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    data |= {"config": config, "status": "stopped"}
    (run_dir / "run.json").write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def resumed(
    served: tuple[TestClient, list[str]], monkeypatch: pytest.MonkeyPatch
) -> tuple[TestClient, list[str]]:
    """Every run reads as resumable, and a resumed job is recorded, not trained."""
    client, _ = served
    queued: list[str] = []

    def record(run_id: str, *_: Any, **kw: Any) -> RunResponse:
        queued.append(kw["profile"].slug)
        return RunResponse(run_id=run_id, status="running")

    monkeypatch.setattr(routes, "_resume_status", lambda run_dir, data: (True, ""))
    monkeypatch.setattr(submission, "_submit_job", record)
    return client, queued


def test_a_stopped_run_on_a_shared_dataset_resumes_for_its_owner(
    resumed: tuple[TestClient, list[str]], world: dict[str, Path]
) -> None:
    client, queued = resumed
    _stopped_run(world, world["shared"] / "beans")
    response = client.post(f"/api/runs/{RUN_ID}/resume", headers=ANA)
    assert response.status_code == 200, response.text
    assert queued == [user_slug("ana@lab.org")]
    assert client.post(f"/api/runs/{RUN_ID}/resume", headers=BIA).status_code == 404


def test_a_run_whose_dataset_left_the_scope_does_not_resume(
    resumed: tuple[TestClient, list[str]], world: dict[str, Path]
) -> None:
    """The stored config is read again like a new one."""
    client, queued = resumed
    _stopped_run(world, world["secret"])
    response = client.post(f"/api/runs/{RUN_ID}/resume", headers=ANA)
    assert response.status_code == 422, response.text
    assert "data.base_dir" in response.json()["detail"]
    assert queued == []


def test_the_session_no_longer_lists_these_actions_as_closed(
    served: tuple[TestClient, list[str]],
) -> None:
    client, _ = served
    closed = client.get("/api/session", headers=ANA).json()["closed"]
    for action in (*ACTIONS, "resume"):
        assert f"POST /api/runs/{{run_id}}/{action}" not in closed
    assert "POST /api/runs/{run_id}/mlflow/retry" in closed
