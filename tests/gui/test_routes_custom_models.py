"""GET /api/model/custom: the models of ``user_models/`` the form can offer (ADR-048).

The classification Model step lists them so ``model.custom_model`` can be picked
in the interface instead of only arriving in an imported YAML. The route scans
the folder on each call, like ``build_custom_model`` does on a miss, and on a
shared server it does not answer: it would run code a user brings.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from visionforge.gui.server_mode import ROUTE_CLASSES, ServerMode, closed_routes
from visionforge.models.registry import clear_registry, registered_models

ROUTE = "/api/model/custom"
PROXY = "10.0.0.5"
USER = {"X-Forwarded-User": "ana@lab.org"}


@pytest.fixture(autouse=True)
def _clean_registry() -> Iterator[None]:
    """Keep the module-global registry isolated between tests."""
    clear_registry()
    yield
    clear_registry()


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty working folder: ``user_models/`` resolves under it."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def client() -> TestClient:
    from visionforge.gui.server import app

    return TestClient(app)


def _write_model(folder: Path, stem: str, name: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{stem}.py").write_text(
        "import torch.nn as nn\n"
        "from visionforge.models.registry import register_model\n"
        f'@register_model("{name}")\n'
        "def build(num_outputs):\n"
        "    return nn.Linear(4, num_outputs)\n",
        encoding="utf-8",
    )


def test_lists_the_models_the_folder_registers(
    workspace: Path, client: TestClient
) -> None:
    _write_model(workspace / "user_models", "zeta", "zeta_net")
    _write_model(workspace / "user_models", "alpha", "alpha_net")

    response = client.get(ROUTE)

    assert response.status_code == 200
    assert response.json() == {"models": ["alpha_net", "zeta_net"]}


def test_no_folder_lists_nothing(workspace: Path, client: TestClient) -> None:
    response = client.get(ROUTE)

    assert response.status_code == 200
    assert response.json() == {"models": []}


def test_a_file_dropped_in_later_is_listed_without_a_restart(
    workspace: Path, client: TestClient
) -> None:
    _write_model(workspace / "user_models", "first", "first_net")
    client.get(ROUTE)
    _write_model(workspace / "user_models", "second", "second_net")

    response = client.get(ROUTE)

    assert response.json() == {"models": ["first_net", "second_net"]}


def test_a_broken_file_does_not_hide_the_others(
    workspace: Path, client: TestClient
) -> None:
    folder = workspace / "user_models"
    _write_model(folder, "good", "good_net")
    (folder / "broken.py").write_text("raise RuntimeError('typo')\n", encoding="utf-8")

    response = client.get(ROUTE)

    assert response.status_code == 200
    assert response.json() == {"models": ["good_net"]}


def test_the_route_is_off_on_a_shared_server() -> None:
    assert ROUTE_CLASSES[f"GET {ROUTE}"] == "off"


def test_the_screen_is_told_the_route_is_closed() -> None:
    assert f"GET {ROUTE}" in closed_routes()


@pytest.fixture
def served(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """The real app in server mode, its profiles under ``tmp_path``."""
    from visionforge.gui.api import routes
    from visionforge.gui.server import app

    monkeypatch.setattr(routes, "_PROFILES_DIR", tmp_path / "profiles")
    monkeypatch.setattr(routes, "_MODELS_DIR", tmp_path / "models")
    had = hasattr(app.state, "server_mode")
    saved = getattr(app.state, "server_mode", None)
    app.state.server_mode = ServerMode.from_options(
        enabled=True, trusted_proxies=[PROXY]
    )
    yield app
    if had:
        app.state.server_mode = saved
    else:
        del app.state.server_mode


def test_a_shared_server_imports_no_user_file(workspace: Path, served: Any) -> None:
    _write_model(workspace / "user_models", "mine", "mine_net")
    client = TestClient(served, base_url="http://localhost", client=(PROXY, 50000))

    response = client.get(ROUTE, headers=USER)

    assert response.status_code == 404
    assert registered_models() == []
