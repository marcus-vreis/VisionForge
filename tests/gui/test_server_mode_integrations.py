"""Server mode: a person's own profile, W&B and Comet with their own token (ADR-223).

The integrations are saved in the profile, so on a server a person reads and
changes their own profile and nobody else's. Local MLflow writes a file on the
server's disk that the person could not open, so it stays off there.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from visionforge.gui.api import profile_tokens, routes
from visionforge.gui.api.dataset_scope import DatasetScope, dataset_scope
from visionforge.gui.api.profiles import resolve_profile
from visionforge.gui.server_mode import ROUTE_CLASSES, ServerMode, answers, user_slug

from ..fake_wandb import FakeGraphQL, FakeWandb, install_fake_wandb

PROXY = "10.0.0.5"
ANA = {"X-Forwarded-User": "ana@lab.org"}
BIA = {"X-Forwarded-User": "bia@lab.org"}
RUN_ID = "20261119_170000_000001"
ANA_TOKEN = "example-key-of-ana"
BIA_TOKEN = "example-key-of-bia"


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """The real app in server mode; tokens are kept under ``tmp_path``."""
    from visionforge.gui.server import app

    monkeypatch.setattr(routes, "_PROFILES_DIR", tmp_path / "profiles")
    monkeypatch.setattr(routes, "_MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(profile_tokens, "config_dir", lambda: tmp_path / "config")
    had = hasattr(app.state, "server_mode")
    saved = getattr(app.state, "server_mode", None)
    app.state.server_mode = ServerMode.from_options(
        enabled=True, trusted_proxies=[PROXY]
    )
    yield TestClient(app, base_url="http://localhost", client=(PROXY, 50000))
    if had:
        app.state.server_mode = saved
    else:
        del app.state.server_mode


def _slug(headers: dict[str, str]) -> str:
    return user_slug(headers["X-Forwarded-User"])


def _wandb_on(client: TestClient, headers: dict[str, str], project: str) -> None:
    saved = client.put(
        f"/api/profiles/{_slug(headers)}",
        json={"settings": {"wandb": {"enabled": True, "project": project}}},
        headers=headers,
    )
    assert saved.status_code == 200, saved.text


def _exported_run(tmp_path: Path, headers: dict[str, str]) -> Path:
    """A finished run in that person's folder whose W&B export failed."""
    slug = _slug(headers)
    run_dir = tmp_path / "profiles" / slug / "models" / "exp" / RUN_ID
    run_dir.mkdir(parents=True)
    record = {
        "id": f"exp_{RUN_ID}",
        "status": "completed",
        "timestamp": "2026-11-19T17:00:00",
        "config": {"name": "exp", "task": "binary", "training": {"epochs": 1}},
        "metrics": {"test_accuracy": 0.75},
        "history": [{"epoch": 1, "train_loss": 0.4}],
    }
    (run_dir / "run.json").write_text(json.dumps(record), encoding="utf-8")
    identity = hashlib.sha256(f"{slug}\0{run_dir.resolve()}".encode()).hexdigest()
    (run_dir / "wandb.json").write_text(
        json.dumps(
            {
                "state": "failed",
                "error": "OSError",
                "status": "FAILED",
                "identity": identity,
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def test_the_route_classes() -> None:
    for key in (
        "GET /api/profiles",
        "PUT /api/profiles/{slug}",
        "POST /api/wandb/runs/{run_id}/retry",
        "POST /api/comet/runs/{run_id}/retry",
    ):
        assert ROUTE_CLASSES[key] == "owner"
        assert answers(key)
    # Local MLflow has no reader on a server; nobody creates a profile there.
    assert ROUTE_CLASSES["POST /api/runs/{run_id}/mlflow/retry"] == "off"
    assert ROUTE_CLASSES["POST /api/profiles"] == "off"


def test_a_person_lists_only_their_own_profile(client: TestClient) -> None:
    client.get("/api/session", headers=BIA)  # bia's profile exists too
    listed = client.get("/api/profiles", headers=ANA)
    assert listed.status_code == 200, listed.text
    (mine,) = listed.json()["profiles"]
    assert mine["slug"] == _slug(ANA)
    assert mine["is_default"] is False
    assert _slug(BIA) not in listed.text


def test_a_person_changes_their_own_profile_and_nobody_elses(
    client: TestClient,
) -> None:
    client.get("/api/session", headers=BIA)
    _wandb_on(client, ANA, "lab-runs")
    named = client.put(
        f"/api/profiles/{_slug(ANA)}",
        json={"name": "Ana Lima", "pronoun": "ela"},
        headers=ANA,
    )
    assert named.status_code == 200, named.text
    (mine,) = client.get("/api/profiles", headers=ANA).json()["profiles"]
    assert mine["name"] == "Ana Lima"
    assert mine["settings"]["wandb"] == {
        "enabled": True,
        "project": "lab-runs",
        "entity": "",
    }

    # Bia's profile, and the local install's default one, are not hers to edit.
    for other in (_slug(BIA), "default"):
        refused = client.put(
            f"/api/profiles/{other}",
            json={"settings": {"wandb": {"enabled": True}}},
            headers=ANA,
        )
        assert refused.status_code == 404, refused.text
    (theirs,) = client.get("/api/profiles", headers=BIA).json()["profiles"]
    assert theirs["settings"]["wandb"]["enabled"] is False
    assert theirs["name"] != "Ana Lima"


def test_local_mlflow_cannot_be_turned_on(client: TestClient) -> None:
    refused = client.put(
        f"/api/profiles/{_slug(ANA)}",
        json={"settings": {"mlflow": {"enabled": True}}},
        headers=ANA,
    )
    assert refused.status_code == 422, refused.text
    assert "MLflow" in refused.json()["detail"]
    (mine,) = client.get("/api/profiles", headers=ANA).json()["profiles"]
    assert mine["settings"]["mlflow"]["enabled"] is False
    # Turning it off (a profile carried over from a local install) is allowed.
    off = client.put(
        f"/api/profiles/{_slug(ANA)}",
        json={"settings": {"mlflow": {"enabled": False}}},
        headers=ANA,
    )
    assert off.status_code == 200, off.text
    retry = client.post(f"/api/runs/{RUN_ID}/mlflow/retry", headers=ANA)
    assert retry.status_code == 404


def test_a_profile_with_mlflow_on_trains_without_it_on_a_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profiles = tmp_path / "profiles"
    monkeypatch.setattr(routes, "_PROFILES_DIR", profiles)
    monkeypatch.setattr(routes, "_MODELS_DIR", tmp_path / "models")
    (profiles / "ana" / "models").mkdir(parents=True)
    (profiles / "ana" / "profile.json").write_text(
        json.dumps(
            {"name": "Ana", "slug": "ana", "settings": {"mlflow": {"enabled": True}}}
        ),
        encoding="utf-8",
    )
    profile = resolve_profile("ana", profiles, tmp_path / "models")
    assert profile.settings.mlflow.enabled

    assert len(routes._profile_trackers(profile)) == 1  # local install: MLflow
    with dataset_scope(DatasetScope(readable=(), owner="ana")):
        assert routes._profile_trackers(profile) == ()


def test_a_retry_sends_the_persons_run_with_their_own_token(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fakes: tuple[FakeWandb, FakeGraphQL] = install_fake_wandb(monkeypatch)
    sdk, api = fakes
    for headers, token in ((ANA, ANA_TOKEN), (BIA, BIA_TOKEN)):
        _wandb_on(client, headers, "lab-runs")
        saved = client.put("/api/wandb/token", json={"token": token}, headers=headers)
        assert saved.status_code == 200, saved.text
    run_dir = _exported_run(tmp_path, ANA)

    # The run is ana's: bia does not find it, and nothing is sent.
    foreign = client.post(f"/api/wandb/runs/{RUN_ID}/retry", headers=BIA)
    assert foreign.status_code == 404, foreign.text
    assert sdk.inits == []

    sent = client.post(f"/api/wandb/runs/{RUN_ID}/retry", headers=ANA)
    assert sent.status_code == 200, sent.text
    assert sent.json()["state"] == "finished"
    assert sdk.inits[0]["settings"].values["api_key"] == ANA_TOKEN
    assert list(api.projects) == [("ana", "lab-runs")]
    # W&B's own local files stay in a folder named after her profile.
    assert Path(sdk.inits[0]["dir"]).name == _slug(ANA)
    assert (run_dir / "wandb.json").is_file()


def test_comet_retry_answers_for_the_owner_only(
    client: TestClient, tmp_path: Path
) -> None:
    _exported_run(tmp_path, ANA)
    # Comet is off in both profiles: the owner reaches the handler (409), the
    # other person does not find the run (404).
    mine = client.post(f"/api/comet/runs/{RUN_ID}/retry", headers=ANA)
    assert mine.status_code == 409, mine.text
    theirs = client.post(f"/api/comet/runs/{RUN_ID}/retry", headers=BIA)
    assert theirs.status_code == 404, theirs.text


def test_the_session_lists_what_stays_closed(client: TestClient) -> None:
    closed = client.get("/api/session", headers=ANA).json()["closed"]
    for key in (
        "GET /api/profiles",
        "PUT /api/profiles/{slug}",
        "POST /api/wandb/runs/{run_id}/retry",
        "POST /api/comet/runs/{run_id}/retry",
    ):
        assert key not in closed
    assert "POST /api/runs/{run_id}/mlflow/retry" in closed
    assert "POST /api/profiles" in closed
