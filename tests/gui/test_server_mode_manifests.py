"""Server mode: a file a dataset names reaches only folders in the scope (ADR-214).

A group manifest for the image audit and the image rows of a regression manifest
are paths read from the request or from a CSV inside the scope; neither may
lead the server to read a file outside it, nor tell whether such a file exists.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from visionforge.gui.api import routes
from visionforge.gui.api.dataset_scope import DatasetScope, dataset_scope
from visionforge.gui.api.evaluation_routes import _execute_run_test
from visionforge.gui.api.sample_dataset import create_sample_dataset
from visionforge.gui.api.schemas import RunTestRequest
from visionforge.gui.server_mode import ServerMode
from visionforge.utils.selftest_data import build_regression_dataset

PROXY = "10.0.0.5"
ANA = {"X-Forwarded-User": "ana@lab.org"}


@pytest.fixture
def world(tmp_path: Path) -> dict[str, Path]:
    shared = (tmp_path / "shared").resolve()
    secret = (tmp_path / "secret").resolve()
    shared.mkdir()
    secret.mkdir()
    return {"shared": shared, "secret": secret}


@pytest.fixture
def client(
    world: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    from visionforge.gui.server import app

    monkeypatch.setattr(routes, "_PROFILES_DIR", tmp_path / "profiles")
    monkeypatch.setattr(routes, "_MODELS_DIR", tmp_path / "models")
    had = hasattr(app.state, "server_mode")
    saved = getattr(app.state, "server_mode", None)
    app.state.server_mode = ServerMode.from_options(
        enabled=True, trusted_proxies=[PROXY], datasets_dirs=[world["shared"]]
    )
    yield TestClient(app, base_url="http://localhost", client=(PROXY, 50000))
    if had:
        app.state.server_mode = saved
    else:
        del app.state.server_mode


def test_a_group_manifest_outside_the_dataset_is_refused_alike(
    client: TestClient, world: dict[str, Path]
) -> None:
    beans = create_sample_dataset(world["shared"], "classification").path
    planted = world["secret"] / "groups.csv"
    answers: list[tuple[int, str]] = []
    for exists in (True, False):
        if exists:
            planted.write_text("path,group\n", encoding="utf-8")
        else:
            planted.unlink()
        for manifest in (str(planted), "../../secret/groups.csv"):
            resp = client.post(
                "/api/dataset/audit",
                json={"base_dir": str(beans), "group_manifest": manifest},
                headers=ANA,
            )
            answers.append((resp.status_code, resp.json()["detail"]))
    assert {status for status, _ in answers} == {400}
    # Refused for where it is, before it is opened (load_group_manifest).
    assert all("groups.csv" in detail and str(beans) in detail for _, detail in answers)
    # The same answer whether the planted file exists or not: no oracle.
    assert answers[:2] == answers[2:]


def _regression_run(tmp_path: Path) -> Path:
    """A finished regression run with a random-init resnet18 checkpoint."""
    from visionforge.models.regression_factory import RegressionModelFactory
    from visionforge.utils.regression_config import RegressionModelConfig

    base = build_regression_dataset(tmp_path / "trained_on", size=32, rows=8)
    model = RegressionModelFactory.create(
        RegressionModelConfig(name="resnet18", num_targets=1, pretrained=False)
    )
    run_dir = tmp_path / "models" / "reg" / "20261124_120000_000000"
    checkpoint = run_dir / "best_model.pth"
    checkpoint.parent.mkdir(parents=True)
    torch.save(model.state_dict(), checkpoint)
    config: dict[str, Any] = {
        "name": "reg",
        "task": "regression",
        "model": {"name": "resnet18", "num_targets": 1, "pretrained": False},
        "data": {
            "base_dir": str(base),
            "target_columns": ["target"],
            "image_size": 32,
            "num_workers": 0,
            "pin_memory": False,
        },
        "training": {"epochs": 1, "batch_size": 4},
        "device": {"kind": "cpu"},
    }
    record = {
        "status": "completed",
        "config": config,
        "artifacts": {"model": str(checkpoint)},
        "tests": [],
    }
    (run_dir / "run.json").write_text(json.dumps(record), encoding="utf-8")
    return run_dir


@pytest.mark.parametrize("row", ["absolute", "climbing"])
def test_a_regression_test_reads_no_image_outside_the_scope(
    tmp_path: Path, world: dict[str, Path], row: str
) -> None:
    run_dir = _regression_run(tmp_path)
    holdout = build_regression_dataset(world["shared"] / "holdout", size=32, rows=4)
    stolen = world["secret"] / "stolen.png"
    Image.new("RGB", (32, 32), color=(9, 9, 9)).save(stolen)
    name = str(stolen) if row == "absolute" else "../../../secret/stolen.png"
    manifest = holdout / "test.csv"
    lines = manifest.read_text(encoding="utf-8").splitlines()
    manifest.write_text("\n".join([*lines, f"{name},0.5"]) + "\n", encoding="utf-8")
    assert (holdout / "images" / name).resolve() == stolen

    scope = DatasetScope(readable=(world["shared"],), owner="ana")
    with dataset_scope(scope), pytest.raises(ValueError, match=r"\b6\b") as refused:
        _execute_run_test(run_dir, RunTestRequest(data_dir=str(manifest)))
    assert str(world["secret"]) not in str(refused.value)
    saved = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert saved["tests"] == []
    # The local tool reads the manifest it is given, as before.
    assert _execute_run_test(run_dir, RunTestRequest(data_dir=str(manifest))).metrics
