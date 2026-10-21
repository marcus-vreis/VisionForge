"""System probes keep their HTTP payloads and resolve dependencies per request."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest
from fastapi.testclient import TestClient

from visionforge import __version__
from visionforge.gui.api import routes
from visionforge.gui.server import app
from visionforge.utils.cuda import CUDAInfo


@pytest.mark.parametrize("cpu_count, expected", [(None, 1), (16, 16)])
def test_system_info_uses_the_current_worker_budget(
    monkeypatch: pytest.MonkeyPatch, cpu_count: int | None, expected: int
) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: cpu_count)
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    pools: list[int] = []
    budgets = iter([2, 0])

    def budget(*, loader_pools: int) -> int:
        pools.append(loader_pools)
        return next(budgets)

    monkeypatch.setattr(routes, "worker_budget", budget)
    client = TestClient(app)
    for suggested in (2, 0):
        response = client.get("/api/system/info")
        assert response.status_code == 200
        assert response.json() == {
            "cpu_count": expected,
            "suggested_workers": suggested,
            "platform": "Windows",
            "version": __version__,
        }
    assert pools == [3, 3]


@pytest.mark.parametrize(
    "processor, machine, expected",
    [("Ryzen", "AMD64", "Ryzen"), ("", "AMD64", "AMD64"), ("", "", "CPU")],
)
def test_device_info_cpu_name_fallbacks(
    monkeypatch: pytest.MonkeyPatch, processor: str, machine: str, expected: str
) -> None:
    monkeypatch.setattr(platform, "processor", lambda: processor)
    monkeypatch.setattr(platform, "machine", lambda: machine)
    monkeypatch.setattr(routes, "check_cuda", lambda: CUDAInfo(available=False))

    response = TestClient(app).get("/api/device/info")

    assert response.status_code == 200
    assert response.json() == {
        "cuda_available": False,
        "cuda_version": None,
        "cpu_name": expected,
        "gpus": [],
    }


def test_importing_and_building_the_router_has_no_runtime_side_effects(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            dedent("""\
                import sys

                def refuse_side_effects(event, args):
                    if event in {"os.mkdir", "socket.connect", "subprocess.Popen"}:
                        raise AssertionError(f"Unexpected import side effect: {event}")

                sys.addaudithook(refuse_side_effects)
                from visionforge.gui.api import system_service
                assert "fastapi" not in sys.modules
                from visionforge.gui.api.system_routes import build_system_router

                def forbidden_probe():
                    raise AssertionError("Probe called before a request")

                build_system_router(forbidden_probe, forbidden_probe)
                assert "visionforge.gui.api.routes" not in sys.modules
                assert "visionforge.gui.server" not in sys.modules
                assert "visionforge.core.trainer" not in sys.modules
                assert "torch" not in sys.modules
            """),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert list(tmp_path.iterdir()) == []
