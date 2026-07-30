"""Route semantics for the replicated-comparison endpoints (ADR-061).

The orchestration itself is covered by tests/core/test_replicated_comparison.py;
here the runner is mocked so the tests stay fast and assert what the API
guarantees: every task exposes it, bad variants are rejected *before* paying
for N x M trainings, and the stored report carries the paired matrix.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from .conftest import occupy_queue, release_queue


@pytest.fixture
def app_and_routes():  # type: ignore[return]
    import visionforge.gui.api.routes as routes_mod
    from visionforge.gui.server import app

    return app, routes_mod


def _payload(tmp_path: Path) -> dict:
    base = tmp_path / "ds"
    base.mkdir(parents=True, exist_ok=True)
    return {
        "name": "cmp_run",
        "model": {"name": "resnet18", "num_targets": 1, "pretrained": False},
        "data": {"base_dir": str(base), "target_columns": ["target"]},
        "training": {"epochs": 1, "batch_size": 8, "learning_rate": 0.001},
        "output": {
            "models_dir": str(tmp_path / "models"),
            "reports_dir": str(tmp_path / "reports"),
        },
    }


def _variants() -> dict[str, dict[str, Any]]:
    return {"baseline": {}, "lr_alto": {"training.learning_rate": 0.01}}


def _wait(client: TestClient) -> dict:
    status: dict = {"status": "running"}
    for _ in range(80):
        status = client.get("/api/experiment/status").json()
        if status["status"] in ("completed", "failed"):
            return status
        time.sleep(0.05)
    return status


class TestRoutesExist:
    def test_every_task_exposes_replicated_comparison(
        self, app_and_routes: tuple
    ) -> None:
        # Probe the APIRouter directly: depending on the Starlette version,
        # app.routes may hide included-router children entirely.
        _, routes_mod = app_and_routes
        paths = {route.path for route in routes_mod.router.routes}
        for task in (
            "classification",
            "regression",
            "segmentation",
            "detection",
            "anomaly",
        ):
            assert f"/api/{task}/replicated-comparison" in paths
        assert "/api/custom/{key}/replicated-comparison" in paths


class TestDispatch:
    def test_report_carries_the_paired_matrix(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        app, routes_mod = app_and_routes
        routes_mod._current_run = None

        captured: dict[str, Any] = {}

        def fake_comparison(  # type: ignore[no-untyped-def]
            runner, base, variants, seeds, metric, *, alpha=0.05, progress_callback=None
        ):
            captured["variants"] = variants
            captured["seeds"] = seeds
            captured["metric"] = metric
            captured["alpha"] = alpha
            return {
                "kind": "replicated_comparison",
                "metric": metric,
                "seeds": seeds,
                "alpha": alpha,
                "variants": {k: {"successful": len(seeds)} for k in variants},
                "comparisons": [
                    {
                        "label_a": "baseline",
                        "label_b": "lr_alto",
                        "p_value": 0.01,
                        "significant": True,
                        "underpowered": False,
                    }
                ],
                "best_by_mean": "baseline",
                "ranked_by_mean": ["baseline", "lr_alto"],
                "significant_pairs": 1,
                "skipped_variants": [],
                "underpowered": False,
                "power_note": "",
            }

        orig = routes_mod.run_replicated_comparison
        routes_mod.run_replicated_comparison = fake_comparison
        try:
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.post(
                "/api/regression/replicated-comparison",
                json={
                    "config": _payload(tmp_path),
                    "variants": _variants(),
                    "seeds": [1, 2, 3],
                    "metric": "r2",
                },
            )
            assert resp.status_code == 200, resp.text
            run_id = resp.json()["run_id"]
            assert _wait(client)["status"] == "completed"

            assert captured["seeds"] == [1, 2, 3]
            assert captured["metric"] == "r2"
            assert set(captured["variants"]) == {"baseline", "lr_alto"}

            report = client.get(f"/api/experiment/result/{run_id}").json()["report"]
            assert report["kind"] == "replicated_comparison"
            assert report["best_by_mean"] == "baseline"
            assert report["significant_pairs"] == 1
            # the durable artifacts, including the paper-ready table
            report_dir = Path(report["report_dir"])
            assert (report_dir / "comparison_summary.json").exists()
            assert (report_dir / "comparison_table.tex").exists()
        finally:
            routes_mod.run_replicated_comparison = orig
            routes_mod._current_run = None

    def test_seeds_derived_from_the_config_when_omitted(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        app, routes_mod = app_and_routes
        routes_mod._current_run = None
        captured: dict[str, Any] = {}

        def fake_comparison(  # type: ignore[no-untyped-def]
            runner, base, variants, seeds, metric, *, alpha=0.05, progress_callback=None
        ):
            captured["seeds"] = seeds
            return {"comparisons": [{"x": 1}], "skipped_variants": []}

        orig = routes_mod.run_replicated_comparison
        routes_mod.run_replicated_comparison = fake_comparison
        try:
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.post(
                "/api/regression/replicated-comparison",
                json={
                    "config": _payload(tmp_path),
                    "variants": _variants(),
                    "n_replicates": 3,
                },
            )
            assert resp.status_code == 200, resp.text
            _wait(client)
            # RegressionConfig defaults training.seed to 42 -> 42, 43, 44
            assert captured["seeds"] == [42, 43, 44]
        finally:
            routes_mod.run_replicated_comparison = orig
            routes_mod._current_run = None


class TestGuards:
    def test_unknown_override_path_is_rejected_before_training(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        app, routes_mod = app_and_routes
        routes_mod._current_run = None
        client = TestClient(app, raise_server_exceptions=True)
        resp = client.post(
            "/api/regression/replicated-comparison",
            json={
                "config": _payload(tmp_path),
                "variants": {"a": {}, "b": {"training.nope": 1}},
                "seeds": [1, 2],
            },
        )
        assert resp.status_code == 422
        assert "training.nope" in resp.text
        # nothing was started
        assert routes_mod._current_run is None

    def test_single_variant_rejected_by_schema(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        app, routes_mod = app_and_routes
        routes_mod._current_run = None
        client = TestClient(app, raise_server_exceptions=True)
        resp = client.post(
            "/api/regression/replicated-comparison",
            json={"config": _payload(tmp_path), "variants": {"only": {}}},
        )
        assert resp.status_code == 422

    def test_duplicate_seeds_rejected(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        app, routes_mod = app_and_routes
        routes_mod._current_run = None
        client = TestClient(app, raise_server_exceptions=True)
        resp = client.post(
            "/api/regression/replicated-comparison",
            json={
                "config": _payload(tmp_path),
                "variants": _variants(),
                "seeds": [1, 1, 2],
            },
        )
        assert resp.status_code == 422

    def test_queues_behind_a_running_job(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        app, routes_mod = app_and_routes
        occupy_queue(routes_mod)
        try:
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.post(
                "/api/regression/replicated-comparison",
                json={"config": _payload(tmp_path), "variants": _variants()},
            )
            assert resp.status_code == 200
            assert resp.json()["status"] == "queued"
        finally:
            release_queue(routes_mod)

    def test_all_variants_failing_marks_the_run_failed(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        app, routes_mod = app_and_routes
        routes_mod._current_run = None

        def fake_comparison(  # type: ignore[no-untyped-def]
            runner, base, variants, seeds, metric, *, alpha=0.05, progress_callback=None
        ):
            return {"comparisons": [], "skipped_variants": list(variants)}

        orig = routes_mod.run_replicated_comparison
        routes_mod.run_replicated_comparison = fake_comparison
        try:
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.post(
                "/api/regression/replicated-comparison",
                json={
                    "config": _payload(tmp_path),
                    "variants": _variants(),
                    "seeds": [1, 2],
                },
            )
            assert resp.status_code == 200
            status = _wait(client)
            assert status["status"] == "failed"
            # The error names what was skipped instead of a bare traceback.
            assert "baseline" in status["error"]
        finally:
            routes_mod.run_replicated_comparison = orig
            routes_mod._current_run = None
