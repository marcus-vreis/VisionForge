"""Comet routes (ADR-200): profile settings, key, retry and the run detail."""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import visionforge.gui.api.routes as routes_mod
from visionforge.core.run_trackers import export_run_record
from visionforge.gui.api.profiles import resolve_profile
from visionforge.gui.api.tracker_exports import profile_trackers
from visionforge.gui.server import app

from ..fake_comet import FakeComet, install_fake_comet

_RUN_ID = "20261112_120000_000000"
_TOKEN = "example-comet-key"


@pytest.fixture
def outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A throwaway outputs/ and home, wired into the routes."""
    monkeypatch.setenv("VISIONFORGE_HOME", str(tmp_path / "home"))
    profiles = tmp_path / "profiles"
    models = tmp_path / "models"
    profiles.mkdir()
    models.mkdir()
    with (
        patch.object(routes_mod, "_PROFILES_DIR", profiles),
        patch.object(routes_mod, "_MODELS_DIR", models),
    ):
        yield tmp_path


@pytest.fixture
def sdk(monkeypatch: pytest.MonkeyPatch) -> FakeComet:
    return install_fake_comet(monkeypatch)


def _settings(outputs: Path, **comet: Any) -> None:
    meta = {"name": "Ana", "settings": {"comet": {"enabled": True, **comet}}}
    (outputs / "profiles" / "default.json").write_text(
        json.dumps(meta), encoding="utf-8"
    )


def _run(outputs: Path, *, status: str = "FAILED", sidecar: bool = True) -> Path:
    run_dir = outputs / "models" / "exp" / _RUN_ID
    run_dir.mkdir(parents=True)
    record = {
        "id": f"exp_{_RUN_ID}",
        "status": "completed",
        "timestamp": "2026-11-12T12:00:00",
        "config": {
            "name": "exp",
            "task": "binary",
            "training": {"epochs": 2},
            "data": {"base_dir": str(outputs / "private-dataset")},
        },
        "metrics": {"test_accuracy": 0.75},
        "history": [
            {"epoch": 1, "train_loss": 0.4},
            {"epoch": 2, "train_loss": 0.3},
        ],
    }
    (run_dir / "run.json").write_text(json.dumps(record), encoding="utf-8")
    if sidecar:
        (run_dir / "comet.json").write_text(
            json.dumps(
                {
                    "state": "failed",
                    "error": "OSError",
                    "status": status,
                    "identity": _identity(run_dir),
                }
            ),
            encoding="utf-8",
        )
    return run_dir


def _identity(run_dir: Path) -> str:
    # The tracker's own key for this run of the default profile.
    return hashlib.sha256(f"default\0{run_dir.resolve()}".encode()).hexdigest()


def _save_token(client: TestClient, token: str = _TOKEN) -> Any:
    return client.put("/api/comet/token", json={"token": token})


def _retry(client: TestClient) -> Any:
    return client.post(f"/api/comet/runs/{_RUN_ID}/retry")


class TestToken:
    def test_saved_key_is_reported_never_returned(self, outputs: Path) -> None:
        client = TestClient(app)
        assert client.get("/api/comet/token").json()["configured"] is False
        saved = _save_token(client)
        assert saved.status_code == 200
        assert saved.json()["configured"] is True
        assert _TOKEN not in saved.text
        assert client.delete("/api/comet/token").json()["configured"] is False

    def test_wandb_and_comet_keys_are_separate(self, outputs: Path) -> None:
        client = TestClient(app)
        _save_token(client)
        assert client.get("/api/wandb/token").json()["configured"] is False

    @pytest.mark.parametrize("body", [{"token": " "}, {"token": "a b"}, {"x": 1}])
    def test_a_bad_payload_is_refused_without_echoing_it(
        self, outputs: Path, body: dict[str, Any]
    ) -> None:
        response = TestClient(app).put("/api/comet/token", json=body)
        assert response.status_code == 422
        assert "a b" not in response.text


class TestSettings:
    def test_comet_settings_are_saved_without_touching_wandb(
        self, outputs: Path
    ) -> None:
        client = TestClient(app)
        client.put(
            "/api/profiles/default", json={"settings": {"wandb": {"enabled": True}}}
        )
        response = client.put(
            "/api/profiles/default",
            json={"settings": {"comet": {"enabled": True, "project": "lab-runs"}}},
        )
        assert response.status_code == 200
        settings = response.json()["settings"]
        assert settings["wandb"]["enabled"] is True
        assert settings["comet"] == {
            "enabled": True,
            "project": "lab-runs",
            "workspace": "",
        }

    def test_turning_it_off_keeps_the_destination(self, outputs: Path) -> None:
        client = TestClient(app)
        body = {"enabled": True, "project": "lab-runs", "workspace": "lab"}
        client.put("/api/profiles/default", json={"settings": {"comet": body}})
        response = client.put(
            "/api/profiles/default", json={"settings": {"comet": {"enabled": False}}}
        )
        assert response.json()["settings"]["comet"] == {**body, "enabled": False}

    @pytest.mark.parametrize(
        "comet", [{"project": "../x"}, {"project": ""}, {"workspace": "Lab"}]
    )
    def test_invalid_names_are_refused(
        self, outputs: Path, comet: dict[str, Any]
    ) -> None:
        response = TestClient(app).put(
            "/api/profiles/default", json={"settings": {"comet": comet}}
        )
        assert response.status_code == 422


class TestTrackerFactories:
    def test_comet_joins_the_exporters_when_on(self, outputs: Path) -> None:
        _settings(outputs)
        profile = resolve_profile(None, outputs / "profiles", outputs / "models")
        factories = profile_trackers(profile, outputs)
        assert [f.func.__name__ for f in factories] == ["CometTracker"]  # type: ignore[attr-defined]

    def test_off_by_default_nothing_is_imported_or_sent(
        self, outputs: Path, sdk: FakeComet, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_dir = _run(outputs, sidecar=False)
        profile = resolve_profile(None, outputs / "profiles", outputs / "models")
        monkeypatch.delitem(sys.modules, "comet_ml")

        export_run_record(profile_trackers(profile, outputs), run_dir)

        assert "comet_ml" not in sys.modules
        assert sdk.calls == []
        assert sdk.starts == []
        assert not (run_dir / "comet.json").exists()


class TestRetry:
    def test_unknown_run_is_404(self, outputs: Path) -> None:
        _settings(outputs)
        assert _retry(TestClient(app)).status_code == 404

    def test_export_off_is_409(self, outputs: Path) -> None:
        _settings(outputs, enabled=False)
        _run(outputs)
        assert _retry(TestClient(app)).status_code == 409

    def test_a_run_never_exported_is_409(self, outputs: Path) -> None:
        _settings(outputs)
        _run(outputs, sidecar=False)
        assert _retry(TestClient(app)).status_code == 409

    def test_a_run_still_training_is_409(self, outputs: Path) -> None:
        _settings(outputs)
        _run(outputs, status="RUNNING")
        assert _retry(TestClient(app)).status_code == 409

    def test_without_a_key_is_409(self, outputs: Path) -> None:
        _settings(outputs)
        _run(outputs)
        response = _retry(TestClient(app))
        assert response.status_code == 409
        assert "Comet" in response.json()["detail"]

    def test_retry_sends_the_run_and_the_detail_shows_it(
        self, outputs: Path, sdk: FakeComet
    ) -> None:
        _settings(outputs, project="lab-runs")
        run_dir = _run(outputs)
        client = TestClient(app)
        _save_token(client)

        response = _retry(client)

        assert response.status_code == 200, response.text
        status = response.json()
        assert status["state"] == "finished"
        assert status["status"] == "FAILED"
        assert status["url"].startswith("https://")
        assert sdk.projects == {("ana", "lab-runs"): False}
        (experiment,) = sdk.experiments.values()
        assert experiment.steps() == [1, 2, None]
        assert sdk.starts[0]["api_key"] == _TOKEN
        sent = json.dumps([experiment.parameters, experiment.metrics])
        assert "private-dataset" not in sent
        assert str(outputs) not in sent
        detail = client.get(f"/api/runs/{_RUN_ID}").json()
        assert detail["comet"]["url"] == status["url"]
        assert "identity" not in detail["comet"]
        assert "synced_epoch" not in detail["comet"]
        assert (run_dir / "comet.json").is_file()

    def test_a_non_https_link_never_reaches_the_page(self, outputs: Path) -> None:
        _settings(outputs)
        run_dir = _run(outputs)
        (run_dir / "comet.json").write_text(
            json.dumps(
                {
                    "state": "finished",
                    "status": "FINISHED",
                    "url": "javascript:alert(1)",
                }
            ),
            encoding="utf-8",
        )
        detail = TestClient(app).get(f"/api/runs/{_RUN_ID}").json()
        assert detail["comet"] is None
