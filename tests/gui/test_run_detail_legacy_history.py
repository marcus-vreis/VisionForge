"""The run detail serves an old anomaly history under today's names (ADR-241)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from fastapi.testclient import TestClient

_RUN_ID = "20260522_100000_000000"


def _detail(tmp_path: Path, history: list[dict[str, Any]]) -> dict[str, Any]:
    import visionforge.gui.api.routes as routes_mod
    from visionforge.gui.server import app

    run_dir = tmp_path / "exp1" / _RUN_ID
    run_dir.mkdir(parents=True)
    data: dict[str, Any] = {
        "id": f"exp1_{_RUN_ID}",
        "experiment": "exp1",
        "timestamp": "2026-05-22T10:00:00",
        "status": "completed",
        "run_dir": str(run_dir.resolve()),
        "config": {"task": "anomaly"},
        "metrics": {"auroc": 0.9},
        "history": history,
        "artifacts": {},
        "tests": [],
    }
    (run_dir / "run.json").write_text(json.dumps(data), encoding="utf-8")
    with patch.object(routes_mod, "_MODELS_DIR", tmp_path):
        resp = TestClient(app).get(f"/api/runs/{run_dir.name}")
    assert resp.status_code == 200
    body: dict[str, Any] = resp.json()
    return body


def test_an_old_run_json_is_served_with_test_auroc(tmp_path: Path) -> None:
    body = _detail(tmp_path, [{"epoch": 1, "train_loss": 0.2, "val_auroc": 0.9}])

    assert body["history"] == [{"epoch": 1, "train_loss": 0.2, "test_auroc": 0.9}]


def test_the_renamed_series_gets_a_direction(tmp_path: Path) -> None:
    body = _detail(tmp_path, [{"epoch": 1, "val_auroc": 0.9}])

    assert body["metric_directions"]["test_auroc"] == "higher"


def test_the_file_on_disk_is_not_rewritten(tmp_path: Path) -> None:
    _detail(tmp_path, [{"epoch": 1, "val_auroc": 0.9}])

    saved = json.loads((tmp_path / "exp1" / _RUN_ID / "run.json").read_text("utf-8"))
    assert saved["history"] == [{"epoch": 1, "val_auroc": 0.9}]


class _Finished:
    status = "FINISHED"


def test_a_tracker_retry_resends_an_old_run_with_test_auroc(tmp_path: Path) -> None:
    from visionforge.gui.api.tracker_exports import retry_record

    record = {
        "config": {"task": "anomaly"},
        "history": [{"epoch": 1, "val_auroc": 0.9}],
    }
    (tmp_path / "run.json").write_text(json.dumps(record), encoding="utf-8")

    data = retry_record(tmp_path, _Finished(), "wandb.retry_unavailable")

    assert data["history"] == [{"epoch": 1, "test_auroc": 0.9}]
