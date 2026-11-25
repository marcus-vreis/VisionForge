"""Server mode: three routes that read a dataset stay inside the scope.

- ``POST /api/model/defaults`` samples images of ``base_dir``: the folder is
  confined like any dataset path, and the walk runs off the event loop.
- Testing an anomaly run on a folder calibrates on the run's training split,
  found next to the chosen folder: that split must be in the scope too.
- Suspicious labels read the run's stored dataset again, like ``/resume``;
  they write into the run, so they count for the quota and the action limit.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel, ValidationError

from visionforge.gui.action_limit import heavy
from visionforge.gui.api import evaluation_routes, label_suspects_run, routes
from visionforge.gui.api.dataset_scope import DatasetScope, dataset_scope
from visionforge.gui.api.evaluation_routes import _execute_run_test
from visionforge.gui.api.label_suspects_run import (
    LabelSuspectsRequest,
    execute_label_suspects,
)
from visionforge.gui.api.schemas import ModelDefaultsRequest, RunTestRequest
from visionforge.gui.disk_quota import grows
from visionforge.gui.server_mode import ROUTE_CLASSES, ServerMode, answers, user_slug
from visionforge.utils.messages import tr
from visionforge.utils.selftest_data import build_anomaly_dataset

PROXY = "10.0.0.5"
ANA = {"X-Forwarded-User": "ana@lab.org"}
RUN_ID = "20261124_120000_000001"
SUSPECTS = "/api/runs/{run_id}/label_suspects"


@pytest.fixture
def world(tmp_path: Path) -> dict[str, Path]:
    shared = tmp_path / "shared"
    (shared / "beans" / "val" / "a").mkdir(parents=True)
    secret = tmp_path / "secret"
    (secret / "val" / "a").mkdir(parents=True)
    return {"shared": shared.resolve(), "secret": secret.resolve(), "root": tmp_path}


@pytest.fixture
def served(
    world: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[TestClient, list[str]]]:
    """The real app in server mode; suspicious labels are recorded, never run."""
    from visionforge.gui.server import app

    reached: list[str] = []

    def record(run_dir: Path, req: BaseModel) -> Any:
        reached.append("label_suspects")
        raise ValueError("reached label_suspects")

    monkeypatch.setattr(evaluation_routes, "execute_label_suspects", record)
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


# --- POST /api/model/defaults ------------------------------------------------


def test_the_defaults_route_is_confined_and_answers() -> None:
    assert ROUTE_CLASSES["POST /api/model/defaults"] == "confined"
    assert answers("POST /api/model/defaults")


def test_the_defaults_base_dir_must_be_in_the_scope(world: dict[str, Path]) -> None:
    inside, outside = world["shared"] / "beans", world["secret"]
    with dataset_scope(DatasetScope(readable=(world["shared"],))):
        assert ModelDefaultsRequest(architecture="resnet18", base_dir=str(inside))
        assert ModelDefaultsRequest(architecture="resnet18").base_dir is None
        with pytest.raises(ValidationError):
            ModelDefaultsRequest(architecture="resnet18", base_dir=str(outside))
    # The local tool samples any folder.
    assert ModelDefaultsRequest(architecture="resnet18", base_dir=str(outside))


def test_on_a_server_the_defaults_do_not_sample_another_folder(
    served: tuple[TestClient, list[str]], world: dict[str, Path]
) -> None:
    client, _ = served
    outside = client.post(
        "/api/model/defaults",
        json={"architecture": "resnet18", "base_dir": str(world["secret"])},
        headers=ANA,
    )
    assert outside.status_code == 422, outside.text
    inside = client.post(
        "/api/model/defaults",
        json={"architecture": "resnet18", "base_dir": str(world["shared"] / "beans")},
        headers=ANA,
    )
    assert inside.status_code == 200, inside.text


def test_the_defaults_walk_the_folder_off_the_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from visionforge.core import image_size
    from visionforge.gui.server import app

    on_loop: list[bool] = []

    def median(root: Path, *, sample: int = 200) -> int | None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            on_loop.append(False)
        else:
            on_loop.append(True)
        return None

    monkeypatch.setattr(image_size, "median_image_side", median)
    response = TestClient(app).post(
        "/api/model/defaults",
        json={"architecture": "resnet18", "base_dir": str(tmp_path)},
    )
    assert response.status_code == 200, response.text
    assert on_loop
    assert not any(on_loop)


def test_a_folder_that_cannot_be_read_leaves_the_size_unsuggested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from visionforge.core import image_size
    from visionforge.gui.server import app

    def refuse(*_: Any, **__: Any) -> Any:
        raise PermissionError("denied")

    monkeypatch.setattr(image_size, "median_image_side", refuse)
    monkeypatch.setattr(image_size, "suggested_image_size", refuse)
    response = TestClient(app).post(
        "/api/model/defaults",
        json={"architecture": "resnet18", "base_dir": str(tmp_path)},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["image_size"], body["dataset_median_side"]) == (None, None)
    assert body["optimizer"] == "adam"


# --- POST /api/runs/{run_id}/test on an anomaly run --------------------------


def _anomaly_run(root: Path, base: Path) -> Path:
    run_dir = root / "models" / "anom" / RUN_ID
    (run_dir / "weights").mkdir(parents=True)
    checkpoint = run_dir / "weights" / "best.pth"
    checkpoint.write_bytes(b"")
    config = {
        "name": "anom",
        "task": "anomaly",
        "model": {"name": "autoencoder", "latent_dim": 8},
        "data": {"base_dir": str(base), "image_size": 32, "num_workers": 0},
        "training": {"epochs": 1, "batch_size": 2},
        "device": {"kind": "cpu"},
    }
    data = {
        "id": f"anom_{RUN_ID}",
        "status": "completed",
        "config": config,
        "artifacts": {"model": str(checkpoint)},
        "tests": [],
    }
    (run_dir / "run.json").write_text(json.dumps(data), encoding="utf-8")
    return run_dir


def test_an_anomaly_test_never_calibrates_on_a_folder_outside_the_scope(
    tmp_path: Path,
) -> None:
    """The operator shares only ``test``: the ``train/good`` beside it, one
    level above the shared folder, is not the person's to read."""
    base = build_anomaly_dataset(tmp_path / "ds", size=32, normals=2).resolve()
    run_dir = _anomaly_run(tmp_path, base)
    shared_test = base / "test"
    with dataset_scope(DatasetScope(readable=(shared_test,))):
        request = RunTestRequest(data_dir=str(shared_test))
        with pytest.raises(ValueError) as excinfo:
            _execute_run_test(run_dir, request)
    assert str(excinfo.value) == tr("testrun.anomaly_train_outside")
    assert json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["tests"] == []


# --- POST/GET /api/runs/{run_id}/label_suspects -------------------------------


def _classification_run(world: dict[str, Path], data: dict[str, Any]) -> Path:
    models = world["root"] / "profiles" / user_slug("ana@lab.org") / "models"
    run_dir = models / "beans" / RUN_ID
    run_dir.mkdir(parents=True)
    (run_dir / "best_model.pth").write_bytes(b"weights")
    record = {
        "run_id": RUN_ID,
        "status": "completed",
        "class_names": ["a", "b"],
        "config": {
            "name": "beans",
            "task": "multiclass",
            "model": {"name": "resnet18", "num_classes": 2, "pretrained": False},
            "data": data,
        },
        "artifacts": {"model": str(run_dir / "best_model.pth")},
    }
    (run_dir / "run.json").write_text(json.dumps(record), encoding="utf-8")
    return run_dir


def test_suspects_on_a_dataset_in_the_scope_reach_the_executor(
    served: tuple[TestClient, list[str]], world: dict[str, Path]
) -> None:
    client, reached = served
    _classification_run(world, {"base_dir": str(world["shared"] / "beans")})
    response = client.post(
        SUSPECTS.format(run_id=RUN_ID), json={"splits": ["val"]}, headers=ANA
    )
    assert (response.status_code, reached) == (400, ["label_suspects"]), response.text


def test_suspects_on_a_dataset_that_left_the_scope_are_refused(
    served: tuple[TestClient, list[str]], world: dict[str, Path]
) -> None:
    """The stored config is read again like a new one, as ``/resume`` does."""
    client, reached = served
    _classification_run(world, {"base_dir": str(world["secret"])})
    response = client.post(
        SUSPECTS.format(run_id=RUN_ID), json={"splits": ["val"]}, headers=ANA
    )
    assert (response.status_code, reached) == (422, []), response.text
    assert "data.base_dir" in response.json()["detail"]


def test_suspects_count_for_the_quota_and_the_action_limit() -> None:
    post, get = f"POST {SUSPECTS}", f"GET {SUSPECTS}"
    assert grows(post)
    assert heavy(post)
    assert not grows(get)
    assert not heavy(get)


def test_a_run_without_a_dataset_folder_is_a_clear_error(tmp_path: Path) -> None:
    run_dir = tmp_path / RUN_ID
    run_dir.mkdir()
    record = {
        "status": "completed",
        "class_names": ["a", "b"],
        "config": {"name": "beans", "task": "multiclass", "data": {}},
        "artifacts": {"model": str(run_dir / "best_model.pth")},
    }
    (run_dir / "run.json").write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        execute_label_suspects(run_dir, LabelSuspectsRequest())
    assert str(excinfo.value) == tr("suspects.no_base_dir")


def test_a_damaged_report_is_a_clear_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from visionforge.gui.server import app

    run_dir = tmp_path / "models" / "beans" / RUN_ID
    (run_dir / label_suspects_run.OUTPUT_DIR).mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps({"status": "completed", "config": {"name": "beans"}}),
        encoding="utf-8",
    )
    report = run_dir / label_suspects_run.OUTPUT_DIR / label_suspects_run.REPORT_FILE
    report.write_text('{"splits": {', encoding="utf-8")
    monkeypatch.setattr(routes, "_MODELS_DIR", tmp_path / "models")
    response = TestClient(app).get(SUSPECTS.format(run_id=RUN_ID))
    assert response.status_code == 400, response.text
    assert response.json()["detail"] == tr("suspects.report_unreadable", run_id=RUN_ID)
