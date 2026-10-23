"""Unsafe run names are refused by HTTP before a training job is submitted."""

from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from visionforge.gui.api import routes


@pytest.mark.parametrize(
    "endpoint", ["/api/experiment/run", "/api/classification/replicates"]
)
@pytest.mark.parametrize("name", ["../escaped", "..\\escaped", "C:\\escaped"])
def test_http_rejects_traversal_before_submission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, endpoint: str, name: str
) -> None:
    """Real config validation rejects traversal without allowing a training thread."""
    monkeypatch.chdir(tmp_path)

    def forbidden_submission(*args: Any, **kwargs: Any) -> None:
        pytest.fail("An unsafe experiment name reached job submission")

    monkeypatch.setattr(routes, "_submit_job", forbidden_submission)
    app = FastAPI()
    app.include_router(routes.router)
    config = {"name": name, "data": {"base_dir": str(tmp_path)}}
    payload = (
        {"config": config, "seeds": [1, 2]}
        if endpoint.endswith("replicates")
        else config
    )
    with TestClient(app) as client:
        response = client.post(endpoint, json=payload)
    assert response.status_code == 422
    assert "name" in response.text
    assert not list(tmp_path.rglob("run.json"))
