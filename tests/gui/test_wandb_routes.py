"""W&B routes (ADR-172): profile settings, token, retry and the run detail."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import visionforge.gui.api.routes as routes_mod
from visionforge.gui.api.profiles import resolve_profile
from visionforge.gui.server import app
from visionforge.utils.config import ExperimentConfig

from ..fake_wandb import FakeGraphQL, FakeWandb, install_fake_wandb

_RUN_ID = "20261108_120000_000000"
_TOKEN = "example-wandb-key"


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
def fakes(monkeypatch: pytest.MonkeyPatch) -> tuple[FakeWandb, FakeGraphQL]:
    return install_fake_wandb(monkeypatch)


def _settings(outputs: Path, **wandb: Any) -> None:
    meta = {"name": "Ana", "settings": {"wandb": {"enabled": True, **wandb}}}
    (outputs / "profiles" / "default.json").write_text(
        json.dumps(meta), encoding="utf-8"
    )


def _run(outputs: Path, *, status: str = "FAILED", sidecar: bool = True) -> Path:
    run_dir = outputs / "models" / "exp" / _RUN_ID
    run_dir.mkdir(parents=True)
    record = {
        "id": f"exp_{_RUN_ID}",
        "status": "completed",
        "timestamp": "2026-11-08T12:00:00",
        "config": {"name": "exp", "task": "binary", "training": {"epochs": 1}},
        "metrics": {"test_accuracy": 0.75},
        "history": [{"epoch": 1, "train_loss": 0.4}],
    }
    (run_dir / "run.json").write_text(json.dumps(record), encoding="utf-8")
    if sidecar:
        (run_dir / "wandb.json").write_text(
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
    return client.put("/api/wandb/token", json={"token": token})


def _retry(client: TestClient) -> Any:
    return client.post(f"/api/wandb/runs/{_RUN_ID}/retry")


class TestToken:
    def test_saved_token_is_reported_never_returned(self, outputs: Path) -> None:
        client = TestClient(app)
        assert client.get("/api/wandb/token").json()["configured"] is False
        saved = _save_token(client)
        assert saved.status_code == 200
        assert saved.json()["configured"] is True
        assert _TOKEN not in saved.text
        assert client.delete("/api/wandb/token").json()["configured"] is False

    def test_hub_and_wandb_tokens_are_separate(self, outputs: Path) -> None:
        client = TestClient(app)
        _save_token(client)
        assert client.get("/api/hub/token").json()["configured"] is False

    @pytest.mark.parametrize("body", [{"token": " "}, {"token": "a b"}, {"x": 1}])
    def test_a_bad_payload_is_refused_without_echoing_it(
        self, outputs: Path, body: dict[str, Any]
    ) -> None:
        response = TestClient(app).put("/api/wandb/token", json=body)
        assert response.status_code == 422
        assert "a b" not in response.text


class TestSettings:
    def test_wandb_settings_are_saved_without_touching_mlflow(
        self, outputs: Path
    ) -> None:
        client = TestClient(app)
        client.put(
            "/api/profiles/default", json={"settings": {"mlflow": {"enabled": True}}}
        )
        response = client.put(
            "/api/profiles/default",
            json={"settings": {"wandb": {"enabled": True, "project": "lab-runs"}}},
        )
        assert response.status_code == 200
        settings = response.json()["settings"]
        assert settings["mlflow"] == {"enabled": True}
        assert settings["wandb"] == {
            "enabled": True,
            "project": "lab-runs",
            "entity": "",
        }

    def test_turning_it_off_keeps_the_destination(self, outputs: Path) -> None:
        client = TestClient(app)
        body = {"enabled": True, "project": "lab-runs", "entity": "lab"}
        client.put("/api/profiles/default", json={"settings": {"wandb": body}})
        response = client.put(
            "/api/profiles/default", json={"settings": {"wandb": {"enabled": False}}}
        )
        assert response.json()["settings"]["wandb"] == {**body, "enabled": False}

    @pytest.mark.parametrize(
        "wandb", [{"project": "../x"}, {"project": ""}, {"entity": "a b"}]
    )
    def test_invalid_names_are_refused(
        self, outputs: Path, wandb: dict[str, Any]
    ) -> None:
        response = TestClient(app).put(
            "/api/profiles/default", json={"settings": {"wandb": wandb}}
        )
        assert response.status_code == 422


class TestTrackerFactories:
    def _config(self, base: Path, **classification: Any) -> ExperimentConfig:
        return ExperimentConfig.model_validate(
            {
                "name": "exp",
                "block": "classification",
                "data": {"base_dir": str(base)},
                "classification": classification,
            }
        )

    def test_both_exporters_when_both_are_on(self, outputs: Path) -> None:
        meta = {"settings": {"mlflow": {"enabled": True}, "wandb": {"enabled": True}}}
        (outputs / "profiles" / "default.json").write_text(json.dumps(meta))
        profile = resolve_profile(None, outputs / "profiles", outputs / "models")
        factories = routes_mod._tracker_factories(self._config(outputs), profile)
        assert [f.func.__name__ for f in factories] == [  # type: ignore[attr-defined]
            "MlflowTracker",
            "WandbTracker",
        ]

    def test_none_outside_a_classification_training(self, outputs: Path) -> None:
        _settings(outputs)
        profile = resolve_profile(None, outputs / "profiles", outputs / "models")
        config = self._config(outputs, mode="evaluate", checkpoint_path="x.pth")
        assert routes_mod._tracker_factories(config, profile) == ()


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

    def test_without_a_token_is_409(self, outputs: Path) -> None:
        _settings(outputs)
        _run(outputs)
        response = _retry(TestClient(app))
        assert response.status_code == 409
        assert "W&B" in response.json()["detail"]

    def test_retry_sends_the_run_and_the_detail_shows_it(
        self, outputs: Path, fakes: tuple[FakeWandb, FakeGraphQL]
    ) -> None:
        sdk, api = fakes
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
        assert api.projects == {("ana", "lab-runs"): "PRIVATE"}
        (run,) = sdk.runs.values()
        assert run.history == {1: {"loss/train": 0.4}}
        assert sdk.inits[0]["settings"].values["api_key"] == _TOKEN
        detail = client.get(f"/api/runs/{_RUN_ID}").json()
        assert detail["wandb"]["url"] == status["url"]
        assert "identity" not in detail["wandb"]
        assert (run_dir / "wandb.json").is_file()

    def test_a_non_https_link_never_reaches_the_page(self, outputs: Path) -> None:
        _settings(outputs)
        run_dir = _run(outputs)
        (run_dir / "wandb.json").write_text(
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
        assert detail["wandb"] is None
