"""GET /api/custom/{key}/code-tour: the code of a researcher's own task."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from torch import nn

from visionforge.gui.server import app
from visionforge.gui.server_mode import ROUTE_CLASSES
from visionforge.tasks import (
    BaseTaskConfig,
    TaskSpec,
    clear_task_registry,
    register_task,
)

URL = "/api/custom/{key}/code-tour"


@pytest.fixture(autouse=True)
def _clean_registry() -> Iterator[None]:
    clear_task_registry()
    yield
    clear_task_registry()


def _register(key: str = "tourtoy") -> None:
    @register_task(
        key=key,
        label="Tour toy",
        accent="#2dd4bf",
        metrics={"mae": "lower"},
        primary_metric="mae",
    )
    class TourToy(TaskSpec):
        Config = BaseTaskConfig

        def build_model(self, cfg: Any) -> nn.Module:
            return nn.Linear(3, 1)

        def build_loaders(self, cfg: Any):
            return [], [], None

        def compute_loss(self, model: nn.Module, batch: Any, cfg: Any):
            return model(batch).sum()

        def compute_metrics(self, model: nn.Module, loader: Any, cfg: Any):
            return {"mae": 0.0}


def test_a_registered_task_shows_its_own_functions() -> None:
    _register()
    response = TestClient(app).get(URL.format(key="tourtoy"))
    assert response.status_code == 200
    sections = {s["id"]: s for s in response.json()}
    assert set(sections) == {"data", "model", "loss", "validate"}
    assert sections["model"]["name"].endswith("TourToy.build_model")
    assert "return nn.Linear(3, 1)" in sections["model"]["lines"][1]
    assert sections["model"]["library"] is None


def test_the_note_text_follows_the_language() -> None:
    _register()
    client = TestClient(app)
    pt = client.get(URL.format(key="tourtoy"), headers={"X-VF-Lang": "pt"}).json()
    en = client.get(URL.format(key="tourtoy"), headers={"X-VF-Lang": "en"}).json()
    assert pt[0]["notes"][0]["text"] != en[0]["notes"][0]["text"]


def test_an_unknown_task_is_a_404_with_a_message() -> None:
    response = TestClient(app).get(URL.format(key="nope"))
    assert response.status_code == 404
    assert response.json()["detail"]


def test_a_builtin_key_has_no_custom_tour() -> None:
    response = TestClient(app).get(URL.format(key="classification"))
    assert response.status_code == 404


def test_server_mode_turns_the_route_off_like_the_other_custom_routes() -> None:
    route = "GET /api/custom/{key}/code-tour"
    assert ROUTE_CLASSES[route] == "off"
    assert ROUTE_CLASSES["GET /api/custom/{key}/schema"] == "off"
