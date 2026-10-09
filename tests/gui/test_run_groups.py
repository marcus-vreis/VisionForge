"""Replicate groups are History runs, end to end (ADR-113).

Real tiny trainings (regression on the synthetic dataset, one epoch) go through
the actual routes, so what is checked is what a researcher gets: the group the
job writes carries the report's numbers, its seeds are tagged, and the History
endpoints serve it as one entry.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from visionforge.utils.selftest_data import build_regression_dataset


@pytest.fixture
def client_and_routes(tmp_path: Path, monkeypatch):  # type: ignore[no-untyped-def]
    import visionforge.gui.api.routes as routes_mod
    from visionforge.gui.server import app

    monkeypatch.setattr(routes_mod, "_MODELS_DIR", tmp_path / "models")
    routes_mod._current_run = None
    try:
        # A context manager keeps one event loop alive, which the training
        # job (a background task of the request that queued it) needs.
        with TestClient(app, raise_server_exceptions=True) as client:
            yield client, routes_mod
    finally:
        routes_mod._RUN_QUEUE.reset()
        routes_mod._current_run = None
        routes_mod._active_cancel_token = None


def _config(tmp_path: Path) -> dict[str, Any]:
    base = build_regression_dataset(tmp_path / "ds")
    return {
        "name": "grp",
        "model": {"name": "resnet18", "num_targets": 1, "pretrained": False},
        "data": {"base_dir": str(base), "target_columns": ["target"], "num_workers": 0},
        "training": {"epochs": 1, "batch_size": 4},
        "output": {
            "models_dir": str(tmp_path / "models"),
            "reports_dir": str(tmp_path / "reports"),
            "graphics_dir": str(tmp_path / "graphics"),
            "logs_dir": str(tmp_path / "logs"),
        },
    }


def _finish(client: TestClient) -> dict[str, Any]:
    status: dict[str, Any] = {"status": "running"}
    for _ in range(900):
        status = client.get("/api/experiment/status").json()
        if status["status"] in ("completed", "failed"):
            return status
        time.sleep(0.1)
    return status


def _group_json(models: Path, run_id: str) -> dict[str, Any]:
    found = list(models.glob(f"*/{run_id}/run.json"))
    assert len(found) == 1, found
    data: dict[str, Any] = json.loads(found[0].read_text(encoding="utf-8"))
    return data


def _report_file(report_dir: str, name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(
        (Path(report_dir) / name).read_text(encoding="utf-8")
    )
    return data


class TestReplicates:
    def test_a_replicate_set_becomes_one_history_entry(
        self, client_and_routes: tuple, tmp_path: Path
    ) -> None:
        client, _ = client_and_routes
        resp = client.post(
            "/api/regression/replicates",
            json={"config": _config(tmp_path), "seeds": [3, 4], "metric": "r2"},
        )
        assert resp.status_code == 200, resp.text
        run_id = resp.json()["run_id"]
        assert _finish(client)["status"] == "completed"

        report = client.get(f"/api/experiment/result/{run_id}").json()["report"]
        summary = _report_file(report["report_dir"], "replicates_summary.json")
        group_json = _group_json(tmp_path / "models", run_id)

        # The group is the report's numbers, not a second computation of them.
        group = group_json["group"]
        assert group["kind"] == "replicates"
        assert group["aggregates"] == summary["aggregates"]
        assert group["metric"] == "r2"
        assert group["seeds"] == [3, 4]
        assert group["seeds_finished"] == [3, 4]
        assert group["stopped"] is False
        assert group["aggregates"]["r2"]["n"] == 2
        assert group["aggregates"]["r2"]["std_ddof"] == 1
        assert group["report_path"] == str(
            Path(report["report_dir"]) / "replicates_summary.json"
        )
        assert group_json["status"] == "completed"
        assert group_json["stopped"] is False
        assert group_json["metrics"]["total_epochs"] == 2

        # Each seed is a plain run named <name>_s<seed>, tagged with the group.
        names = sorted(c["run_id"] for c in group["children"])
        assert len(names) == 2
        for seed, child in zip((3, 4), group["children"], strict=True):
            run_json = json.loads(
                (Path(child["run_dir"]) / "run.json").read_text(encoding="utf-8")
            )
            assert run_json["experiment"] == f"grp_s{seed}"
            assert run_json["group_id"] == run_id
            assert child["run_id"] == Path(child["run_dir"]).name

    def test_history_lists_the_group_and_tags_its_seeds(
        self, client_and_routes: tuple, tmp_path: Path
    ) -> None:
        client, _ = client_and_routes
        run_id = client.post(
            "/api/regression/replicates",
            json={"config": _config(tmp_path), "seeds": [3, 4]},
        ).json()["run_id"]
        assert _finish(client)["status"] == "completed"

        runs = client.get("/api/runs").json()
        group = next(r for r in runs if r["run_id"] == run_id)
        children = [r for r in runs if r.get("group_id") == run_id]

        assert group["group_id"] is None
        assert group["block"] == "replicates"
        assert group["task"] == "regression"
        assert group["resumable"] is False
        assert group["group"]["kind"] == "replicates"
        assert group["group"]["n_finished"] == 2
        assert sorted(group["group"]["child_ids"]) == sorted(
            r["run_id"] for r in children
        )
        assert len(children) == 2
        # The group's History metric is the mean of the seeds' own r2.
        detail = client.get(f"/api/runs/{run_id}").json()
        full = detail["group"]["aggregates"]["r2"]
        assert group["final_metrics"]["r2"] == pytest.approx(full["mean"])
        # The card prints the mean beside the interval it came with, keyed by
        # the same label as the number.
        finals = group["group"]["final_aggregates"]
        assert set(finals) == set(group["final_metrics"])
        assert finals["r2"] == full
        for label, value in group["final_metrics"].items():
            assert finals[label]["mean"] == pytest.approx(value)
        # A seed is still a run of its own, openable and not a group itself.
        assert all(r["group"] is None for r in children)

        assert detail["group"]["metric_keys"] == {
            "test_r2": "r2",
            "test_rmse": "rmse",
            "test_mae": "mae",
            "test_mse": "mse",
        }
        assert detail["group_id"] is None
        assert detail["task"] == "regression"
        child = client.get(f"/api/runs/{children[0]['run_id']}").json()
        assert child["group_id"] == run_id
        assert child["group"] is None

    def test_a_stopped_replicate_set_says_stopped(
        self, client_and_routes: tuple, tmp_path: Path, monkeypatch
    ) -> None:
        client, routes_mod = client_and_routes
        real = routes_mod.run_replicates

        def stop_first(runner, base, seeds, metric, progress_callback=None):  # type: ignore[no-untyped-def]
            # The stop pressed while the first seed is still loading its data.
            routes_mod._active_cancel_token.cancel()
            return real(runner, base, seeds, metric, progress_callback)

        monkeypatch.setattr(routes_mod, "run_replicates", stop_first)
        run_id = client.post(
            "/api/regression/replicates",
            json={"config": _config(tmp_path), "seeds": [3, 4, 5]},
        ).json()["run_id"]
        assert _finish(client)["status"] == "completed"

        group_json = _group_json(tmp_path / "models", run_id)
        assert group_json["stopped"] is True
        group = group_json["group"]
        assert group["stopped"] is True
        assert group["seeds"] == [3, 4, 5]
        assert group["seeds_finished"] == []
        assert [c["status"] for c in group["children"]] == ["stopped"]

        listed = next(
            r for r in client.get("/api/runs").json() if r["run_id"] == run_id
        )
        assert listed["group"]["stopped"] is True


class TestReplicatedComparison:
    def test_a_comparison_becomes_one_history_entry_with_its_paired_tests(
        self, client_and_routes: tuple, tmp_path: Path
    ) -> None:
        client, _ = client_and_routes
        resp = client.post(
            "/api/regression/replicated-comparison",
            json={
                "config": _config(tmp_path),
                "variants": {"base": {}, "lr": {"training.learning_rate": 0.01}},
                "seeds": [3, 4],
                "metric": "r2",
            },
        )
        assert resp.status_code == 200, resp.text
        run_id = resp.json()["run_id"]
        assert _finish(client)["status"] == "completed"

        report = client.get(f"/api/experiment/result/{run_id}").json()["report"]
        summary = _report_file(report["report_dir"], "comparison_summary.json")
        group_json = _group_json(tmp_path / "models", run_id)
        group = group_json["group"]

        assert group["kind"] == "replicated_comparison"
        assert group["comparisons"] == summary["comparisons"]
        assert group["best_by_mean"] == summary["best_by_mean"]
        assert group["ranking_seeds"] == summary["ranking_seeds"]
        assert group["not_run"] == summary["not_run"] == []
        assert set(group["variants"]) == {"base", "lr"}
        for label, variant in group["variants"].items():
            assert variant["aggregates"] == summary["variants"][label]["aggregates"]
            assert variant["seeds_finished"] == [3, 4]
        pair = group["comparisons"][0]
        assert isinstance(pair["p_value"], float)
        assert isinstance(pair["significant"], bool)
        assert group["report_path"].endswith("comparison_summary.json")
        assert (group["n_requested"], group["n_finished"]) == (4, 4)

        # Four seeds' runs, every one tagged with the group.
        tagged = [
            json.loads(p.read_text(encoding="utf-8"))
            for p in (tmp_path / "models").glob("grp_*_s*/*/run.json")
        ]
        assert len(tagged) == 4
        assert {r["group_id"] for r in tagged} == {run_id}

        listed = next(
            r for r in client.get("/api/runs").json() if r["run_id"] == run_id
        )
        assert listed["block"] == "replicated_comparison"
        assert listed["group"]["variants"] == ["base", "lr"]
        assert listed["group"]["best_by_mean"] == summary["best_by_mean"]
        assert len(listed["group"]["child_ids"]) == 4
        # No headline number: a comparison does not have one.
        assert listed["final_metrics"] == {}

        detail = client.get(f"/api/runs/{run_id}").json()
        assert detail["group"]["comparisons"] == summary["comparisons"]
