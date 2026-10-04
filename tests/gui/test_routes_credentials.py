"""POST /api/credentials — storing a provider key never echoes it back."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    # The key file lives under VISIONFORGE_HOME; never touch the real one.
    monkeypatch.setenv("VISIONFORGE_HOME", str(tmp_path / "home"))
    from visionforge.gui.server import app

    return TestClient(app)


class TestSaveCredential:
    def test_a_saved_key_comes_back_masked(
        self, tmp_path: Path, client: TestClient
    ) -> None:
        resp = client.post(
            "/api/credentials",
            json={"provider": "huggingface", "value": "hf_abcdefgh1234"},
        )

        assert resp.status_code == 200
        body = resp.json()
        entry = body["providers"]["huggingface"]
        assert entry["saved"] is True
        assert entry["masked"].endswith("1234")
        assert "hf_abcdefgh" not in resp.text
        assert body["config_dir"] == str(tmp_path / "home")

    def test_the_other_providers_stay_empty(self, client: TestClient) -> None:
        body = client.post(
            "/api/credentials", json={"provider": "kaggle", "value": "KGAT_x1y2"}
        ).json()

        assert body["providers"]["roboflow"] == {"saved": False, "masked": ""}

    def test_a_blank_value_is_a_400(self, client: TestClient) -> None:
        # Passes the schema (min_length=1) and is rejected after strip().
        resp = client.post(
            "/api/credentials", json={"provider": "roboflow", "value": "   "}
        )

        assert resp.status_code == 400

    def test_an_unknown_provider_is_rejected_by_the_schema(
        self, client: TestClient
    ) -> None:
        resp = client.post(
            "/api/credentials", json={"provider": "dropbox", "value": "abc"}
        )

        assert resp.status_code == 422
