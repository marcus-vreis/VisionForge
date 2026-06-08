from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def app_and_routes():  # type: ignore[return]
    import visionforge.gui.api.routes as routes_mod
    from visionforge.gui.server import app

    return app, routes_mod


def _reg_request(tmp_path: Path, model_names: list[str]) -> dict[str, Any]:
    base = tmp_path / "ds"
    base.mkdir(parents=True, exist_ok=True)
    return {
        "config": {
            "name": "reg_cmp",
            "model": {"name": "resnet18", "num_targets": 1, "pretrained": False},
            "data": {"base_dir": str(base), "target_columns": ["target"]},
            "training": {"epochs": 1, "batch_size": 8, "learning_rate": 0.001},
            "output": {
                "models_dir": str(tmp_path / "models"),
                "reports_dir": str(tmp_path / "reports"),
            },
        },
        "model_names": model_names,
        "metric": "r2",
    }


def _seg_request(tmp_path: Path, model_names: list[str]) -> dict[str, Any]:
    base = tmp_path / "ds"
    base.mkdir(parents=True, exist_ok=True)
    return {
        "config": {
            "name": "seg_cmp",
            "model": {
                "name": "deeplabv3_resnet50",
                "num_classes": 2,
                "pretrained": False,
            },
            "data": {"base_dir": str(base)},
            "training": {"epochs": 1, "batch_size": 2, "learning_rate": 0.001},
            "output": {
                "models_dir": str(tmp_path / "models"),
                "reports_dir": str(tmp_path / "reports"),
            },
        },
        "model_names": model_names,
        "metric": "miou",
    }


# ── route surface (dispatch / validation / conflict) ────────────────────────────
# Background runs are not polled to completion through TestClient — the portal
# loop doesn't reliably resume to_thread continuations (the same reason the other
# GUI tests assert dispatch only). Completion is exercised on the executor below.


class TestRegressionCompareRoute:
    def test_accepts_valid_request(self, app_and_routes: tuple, tmp_path: Path) -> None:
        app, routes_mod = app_and_routes
        routes_mod._current_run = {"run_id": "x", "status": "completed"}
        try:
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.post(
                "/api/regression/compare",
                json=_reg_request(tmp_path, ["resnet18", "resnet34"]),
            )
            assert resp.status_code == 200, resp.text
            assert resp.json()["run_id"].startswith("reg_cmp_")
        finally:
            routes_mod._current_run = None

    def test_rejects_single_model(self, app_and_routes: tuple, tmp_path: Path) -> None:
        app, routes_mod = app_and_routes
        routes_mod._current_run = None
        client = TestClient(app, raise_server_exceptions=True)
        resp = client.post(
            "/api/regression/compare", json=_reg_request(tmp_path, ["resnet18"])
        )
        assert resp.status_code == 422

    def test_rejects_unknown_backbone(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        app, routes_mod = app_and_routes
        routes_mod._current_run = None
        client = TestClient(app, raise_server_exceptions=True)
        resp = client.post(
            "/api/regression/compare",
            json=_reg_request(tmp_path, ["resnet18", "not_a_model"]),
        )
        assert resp.status_code == 422

    def test_conflict_when_running(self, app_and_routes: tuple, tmp_path: Path) -> None:
        app, routes_mod = app_and_routes
        routes_mod._current_run = {"run_id": "x", "status": "running"}
        try:
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.post(
                "/api/regression/compare",
                json=_reg_request(tmp_path, ["resnet18", "resnet34"]),
            )
            assert resp.status_code == 409
        finally:
            routes_mod._current_run = None


class TestSegmentationCompareRoute:
    def test_accepts_valid_request(self, app_and_routes: tuple, tmp_path: Path) -> None:
        app, routes_mod = app_and_routes
        routes_mod._current_run = {"run_id": "x", "status": "completed"}
        try:
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.post(
                "/api/segmentation/compare",
                json=_seg_request(tmp_path, ["deeplabv3_resnet50", "fcn_resnet50"]),
            )
            assert resp.status_code == 200, resp.text
        finally:
            routes_mod._current_run = None

    def test_rejects_unknown_model(self, app_and_routes: tuple, tmp_path: Path) -> None:
        app, routes_mod = app_and_routes
        routes_mod._current_run = None
        client = TestClient(app, raise_server_exceptions=True)
        resp = client.post(
            "/api/segmentation/compare",
            json=_seg_request(tmp_path, ["deeplabv3_resnet50", "nope"]),
        )
        assert resp.status_code == 422


# ── executor (background completion, real asyncio loop) ─────────────────────────


def _run(coro: Any) -> None:
    asyncio.run(coro)


class TestRegressionCompareExecutor:
    def test_completes_with_ranked_report(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        from visionforge.gui.api.comparison import RegressionCompareRequest

        _, routes_mod = app_and_routes
        routes_mod._event_queue = asyncio.Queue()
        routes_mod._current_run = {"run_id": "rid", "status": "running"}

        report_by_arch = {"resnet18": 0.7, "resnet34": 0.9}
        seen: list[str] = []

        def fake_setup(self: Any, cfg: Any) -> None:
            seen.append(cfg.model.name)

        def fake_report(self: Any) -> dict[str, Any]:
            return {
                "test": {
                    "mse": 0.1,
                    "rmse": 0.3,
                    "mae": 0.2,
                    "r2": report_by_arch[seen[-1]],
                }
            }

        with (
            patch(
                "visionforge.blocks.regression_runner.RegressionBlock.setup", fake_setup
            ),
            patch("visionforge.blocks.regression_runner.RegressionBlock.run"),
            patch(
                "visionforge.blocks.regression_runner.RegressionBlock.report",
                fake_report,
            ),
        ):
            req = RegressionCompareRequest.model_validate(
                _reg_request(tmp_path, ["resnet18", "resnet34"])
            )
            _run(routes_mod._execute_regression_comparison(req, "rid"))

        run = routes_mod._current_run
        assert run["status"] == "completed", run
        report = run["report"]
        assert report["total_ran"] == 2
        assert report["top_3"][0]["model_arch"] == "resnet34"
        # artifacts landed under reports_dir / name
        assert (tmp_path / "reports" / "reg_cmp" / "ranking.csv").exists()
        routes_mod._current_run = None

    def test_all_failed_marks_run_failed(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        from visionforge.gui.api.comparison import RegressionCompareRequest

        _, routes_mod = app_and_routes
        routes_mod._event_queue = asyncio.Queue()
        routes_mod._current_run = {"run_id": "rid", "status": "running"}

        with (
            patch("visionforge.blocks.regression_runner.RegressionBlock.setup"),
            patch(
                "visionforge.blocks.regression_runner.RegressionBlock.run",
                side_effect=RuntimeError("boom"),
            ),
        ):
            req = RegressionCompareRequest.model_validate(
                _reg_request(tmp_path, ["resnet18", "resnet34"])
            )
            _run(routes_mod._execute_regression_comparison(req, "rid"))

        assert routes_mod._current_run["status"] == "failed"
        routes_mod._current_run = None


class TestSegmentationCompareExecutor:
    def test_completes_with_ranked_report(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        from visionforge.gui.api.comparison import SegmentationCompareRequest

        _, routes_mod = app_and_routes
        routes_mod._event_queue = asyncio.Queue()
        routes_mod._current_run = {"run_id": "rid", "status": "running"}

        miou_by_arch = {"deeplabv3_resnet50": 0.6, "fcn_resnet50": 0.8}
        seen: list[str] = []

        def fake_setup(self: Any, cfg: Any) -> None:
            seen.append(cfg.model.name)

        def fake_report(self: Any) -> dict[str, Any]:
            return {
                "test": {
                    "miou": miou_by_arch[seen[-1]],
                    "dice": 0.8,
                    "pixel_acc": 0.95,
                }
            }

        with (
            patch(
                "visionforge.blocks.segmentation_runner.SegmentationBlock.setup",
                fake_setup,
            ),
            patch("visionforge.blocks.segmentation_runner.SegmentationBlock.run"),
            patch(
                "visionforge.blocks.segmentation_runner.SegmentationBlock.report",
                fake_report,
            ),
        ):
            req = SegmentationCompareRequest.model_validate(
                _seg_request(tmp_path, ["deeplabv3_resnet50", "fcn_resnet50"])
            )
            _run(routes_mod._execute_segmentation_comparison(req, "rid"))

        run = routes_mod._current_run
        assert run["status"] == "completed", run
        assert run["report"]["top_3"][0]["model_arch"] == "fcn_resnet50"
        routes_mod._current_run = None
