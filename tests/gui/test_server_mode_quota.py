"""Server mode: a ceiling on what each person's folder may hold (ADR-225).

``--quota-gb`` is checked when an action that writes new data is asked for.
Past it the person still reads, opens and deletes their runs; deleting is how
they get under it again.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from visionforge.gui.api import routes
from visionforge.gui.disk_quota import GIB, DiskUsage, folder_bytes, grows, size_text
from visionforge.gui.server_mode import (
    ROUTE_CLASSES,
    ServerMode,
    ServerModeError,
    answers,
    user_slug,
)

PROXY = "10.0.0.5"
ANA = {"X-Forwarded-User": "ana@lab.org"}
BIA = {"X-Forwarded-User": "bia@lab.org"}
RUN_ID = "20261120_090000_000001"
MIB = 1024 * 1024


def test_the_option_needs_server_mode_and_a_size_above_zero() -> None:
    with pytest.raises(ServerModeError, match="--quota-gb"):
        ServerMode.from_options(enabled=False, quota_gb=5)
    for bad in (0, -1, float("nan")):
        with pytest.raises(ServerModeError, match="above zero"):
            ServerMode.from_options(enabled=True, trusted_proxies=[PROXY], quota_gb=bad)
    mode = ServerMode.from_options(enabled=True, trusted_proxies=[PROXY], quota_gb=2.5)
    assert mode.quota_bytes == int(2.5 * GIB)
    unlimited = ServerMode.from_options(enabled=True, trusted_proxies=[PROXY])
    assert unlimited.quota_bytes is None


def test_which_routes_write_new_data() -> None:
    growing = {key for key in ROUTE_CLASSES if answers(key) and grows(key)}
    # Every way a training starts, in the five tasks, and a resumed run.
    for key in (
        "POST /api/experiment/run",
        "POST /api/detection/run",
        "POST /api/segmentation/sweep",
        "POST /api/regression/cv",
        "POST /api/anomaly/compare",
        "POST /api/classification/replicates",
        "POST /api/detection/replicated-comparison",
        "POST /api/runs/{run_id}/resume",
    ):
        assert key in growing
    # A dataset that is downloaded, made or derived, and what an evaluation
    # action leaves in the run.
    for key in (
        "POST /api/dataset/download",
        "POST /api/sample-dataset",
        "POST /api/explorer/resplit/{job_id}/apply",
        "POST /api/explorer/coco/import/{job_id}/apply",
        "POST /api/runs/{run_id}/test",
        "POST /api/runs/{run_id}/gradcam",
        "POST /api/runs/{run_id}/batch_predict",
        "POST /api/runs/{run_id}/export_onnx",
        "POST /api/runs/{run_id}/label_suspects",
    ):
        assert key in growing
    # A new route that answers on a server changes this number: decide whether
    # it writes new data before changing it.
    assert len(growing) == 38
    # Reading, planning, cancelling, sending elsewhere and deleting never count.
    for key in (
        "GET /api/runs",
        "DELETE /api/runs/{run_id}",
        "POST /api/dataset/stats",
        "POST /api/explorer/resplit/{job_id}/plan",
        "POST /api/explorer/resplit/{job_id}/cancel",
        "POST /api/dataset/preview_augment",
        "POST /api/hub/export",
        "PUT /api/profiles/{slug}",
    ):
        assert not grows(key)


def test_folder_bytes_adds_every_file_under_the_folder(tmp_path: Path) -> None:
    (tmp_path / "a.bin").write_bytes(b"x" * 1000)
    (tmp_path / "runs" / "one").mkdir(parents=True)
    (tmp_path / "runs" / "one" / "model.pth").write_bytes(b"x" * 2500)
    (tmp_path / "runs" / "empty").mkdir()
    assert folder_bytes(tmp_path) == 3500
    assert folder_bytes(tmp_path / "missing") == 0


def test_a_link_out_of_the_folder_is_not_counted(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "big.bin").write_bytes(b"x" * 9000)
    mine = tmp_path / "mine"
    mine.mkdir()
    (mine / "small.bin").write_bytes(b"x" * 10)
    try:
        (mine / "link").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("this account cannot create symbolic links")
    assert folder_bytes(mine) == 10


def test_a_figure_is_kept_for_a_while_and_dropped_on_request(tmp_path: Path) -> None:
    usage = DiskUsage(stale_after_s=3600)
    (tmp_path / "a.bin").write_bytes(b"x" * 100)
    assert usage.of(tmp_path) == 100
    (tmp_path / "b.bin").write_bytes(b"x" * 50)
    assert usage.of(tmp_path) == 100  # remembered
    usage.forget(tmp_path)
    assert usage.of(tmp_path) == 150
    assert DiskUsage(stale_after_s=0).of(tmp_path) == 150


def test_sizes_are_written_in_the_languages_notation() -> None:
    assert size_text(int(1.5 * GIB), "pt") == "1,5 GB"
    assert size_text(int(1.5 * GIB), "en") == "1.5 GB"
    assert size_text(0, "pt") == "0,0 GB"


def _client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, quota_gb: float | None):
    from visionforge.gui.server import app

    monkeypatch.setattr(routes, "_PROFILES_DIR", tmp_path / "profiles")
    monkeypatch.setattr(routes, "_MODELS_DIR", tmp_path / "models")
    had = hasattr(app.state, "server_mode")
    saved = getattr(app.state, "server_mode", None)
    app.state.server_mode = ServerMode.from_options(
        enabled=True, trusted_proxies=[PROXY], quota_gb=quota_gb
    )
    yield TestClient(app, base_url="http://localhost", client=(PROXY, 50000))
    if had:
        app.state.server_mode = saved
    else:
        del app.state.server_mode


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """The real app in server mode, one mebibyte per person."""
    yield from _client(tmp_path, monkeypatch, 1 / 1024)


@pytest.fixture
def unlimited(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    yield from _client(tmp_path, monkeypatch, None)


def _profile(tmp_path: Path, headers: dict[str, str]) -> Path:
    return tmp_path / "profiles" / user_slug(headers["X-Forwarded-User"])


def _heavy_run(client: TestClient, tmp_path: Path, headers: dict[str, str]) -> Path:
    """A finished run of two mebibytes in that person's folder."""
    client.get("/api/profiles", headers=headers)  # makes the profile, measures nothing
    run_dir = _profile(tmp_path, headers) / "models" / "beans" / RUN_ID
    run_dir.mkdir(parents=True)
    (run_dir / "best_model.pth").write_bytes(b"\0" * (2 * MIB))
    record = {
        "id": f"beans_{RUN_ID}",
        "experiment": "beans",
        "status": "completed",
        "timestamp": "2026-11-20T09:00:00",
        "config": {"name": "beans", "task": "binary", "training": {"epochs": 1}},
        "metrics": {"test_accuracy": 0.75, "total_epochs": 1},
        "history": [{"epoch": 1, "train_loss": 0.4}],
    }
    (run_dir / "run.json").write_text(json.dumps(record), encoding="utf-8")
    return run_dir


def test_a_full_folder_refuses_what_writes_and_keeps_what_reads(
    client: TestClient, tmp_path: Path
) -> None:
    _heavy_run(client, tmp_path, ANA)

    refused = client.post(
        "/api/sample-dataset", json={"task": "classification"}, headers=ANA
    )
    assert refused.status_code == 507, refused.text
    detail = refused.json()["detail"]
    assert "2,0 GB" not in detail  # two mebibytes, not gigabytes
    assert "0,0 GB" in detail  # of a ceiling this small
    assert "Histórico" in detail
    english = client.post(
        "/api/sample-dataset",
        json={"task": "classification"},
        headers={**ANA, "X-VF-Lang": "en"},
    )
    assert "History" in english.json()["detail"]
    # Nothing was written.
    assert not any((_profile(tmp_path, ANA) / "datasets").iterdir())
    # A training is refused before its body is even read.
    assert client.post("/api/experiment/run", json={}, headers=ANA).status_code == 507

    # She still sees and opens her runs, and reads her profile.
    listed = client.get("/api/runs", headers=ANA)
    assert listed.status_code == 200
    assert [run["run_id"] for run in listed.json()] == [RUN_ID]
    assert client.get(f"/api/runs/{RUN_ID}", headers=ANA).status_code == 200
    assert client.get("/api/profiles", headers=ANA).status_code == 200


def test_the_ceiling_is_each_persons_own(client: TestClient, tmp_path: Path) -> None:
    _heavy_run(client, tmp_path, ANA)
    made = client.post(
        "/api/sample-dataset", json={"task": "classification"}, headers=BIA
    )
    assert made.status_code == 201, made.text


def test_deleting_a_run_frees_the_folder_at_once(
    client: TestClient, tmp_path: Path
) -> None:
    run_dir = _heavy_run(client, tmp_path, ANA)
    body = {"task": "classification"}
    assert client.post("/api/sample-dataset", json=body, headers=ANA).status_code == 507

    deleted = client.delete(f"/api/runs/{RUN_ID}", headers=ANA)
    assert deleted.status_code in (200, 204), deleted.text
    assert not run_dir.exists()

    made = client.post("/api/sample-dataset", json=body, headers=ANA)
    assert made.status_code == 201, made.text


def test_the_session_says_how_full_the_folder_is(
    client: TestClient, tmp_path: Path
) -> None:
    _heavy_run(client, tmp_path, ANA)
    client.delete("/api/runs/nothing", headers=ANA)  # drops the remembered figure
    quota = client.get("/api/session", headers=ANA).json()["quota"]
    assert quota["limit_bytes"] == MIB
    assert quota["used_bytes"] >= 2 * MIB
    assert quota["used_bytes"] == folder_bytes(_profile(tmp_path, ANA))


def test_without_the_option_nothing_is_measured_or_refused(
    unlimited: TestClient, tmp_path: Path
) -> None:
    _heavy_run(unlimited, tmp_path, ANA)
    assert unlimited.get("/api/session", headers=ANA).json()["quota"] is None
    made = unlimited.post(
        "/api/sample-dataset", json={"task": "classification"}, headers=ANA
    )
    assert made.status_code == 201, made.text
