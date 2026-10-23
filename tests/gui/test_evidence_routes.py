"""Evidence endpoints use real files and respect the active profile."""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from PIL import Image

from visionforge.core.experiment_bundle import verify_experiment_bundle
from visionforge.gui.api import evidence_routes, routes
from visionforge.gui.api.profiles import default_profile
from visionforge.gui.server import app

from .conftest import occupy_queue, release_queue


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(routes, "_MODELS_DIR", Path("outputs/models"))
    monkeypatch.setattr(routes, "_PROFILES_DIR", Path("outputs/profiles"))
    return TestClient(app)


def _run(root: Path) -> Path:
    run = root / "example" / "run"
    run.mkdir(parents=True)
    (run / "config.json").write_text("{}")
    (run / "run.json").write_text(
        json.dumps({"id": "example_run", "artifacts": {"config": "config.json"}}),
        encoding="utf-8",
    )
    return run


@pytest.mark.parametrize(
    ("path", "method", "status_codes"),
    [
        ("/api/dataset/audit", "post", {"400"}),
        ("/api/runs/{run_id}/bundle", "get", {"400", "404", "409"}),
    ],
)
def test_openapi_documents_evidence_errors(
    client: TestClient, path: str, method: str, status_codes: set[str]
) -> None:
    """Evidence clients can discover every explicit error response in OpenAPI."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    responses = response.json()["paths"][path][method]["responses"]
    assert status_codes <= responses.keys()


def test_audit_exposes_findings_without_rewriting_inputs(
    client: TestClient, tmp_path: Path
) -> None:
    dataset = tmp_path / "data"
    for split in ("train", "val"):
        folder = dataset / split
        folder.mkdir(parents=True)
        Image.new("RGB", (8, 8), "red").save(folder / "sample.png")
    before = {path: path.read_bytes() for path in dataset.rglob("*.png")}
    response = client.post(
        "/api/dataset/audit",
        json={"base_dir": str(dataset), "splits": {"train": "train", "val": "val"}},
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["ok"] is False
    assert result["image_counts"] == {"train": 1, "val": 1}
    assert len(result["duplicate_groups"]) == 1
    assert result["visual_similarity_checked"] is False
    assert all(path.read_bytes() == content for path, content in before.items())


def test_bundle_is_a_verifiable_download(client: TestClient, tmp_path: Path) -> None:
    _run(Path("outputs/models"))
    response = client.get("/api/runs/example_run/bundle")
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/zip"
    assert "attachment" in response.headers["content-disposition"]
    downloaded = tmp_path / "download.zip"
    downloaded.write_bytes(response.content)
    assert set(verify_experiment_bundle(downloaded)["files"]) == {
        "run.json",
        "config.json",
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"base_dir": ""},
        {"base_dir": ".", "strict_groups": True},
        {"base_dir": ".", "splits": {"train": "../private"}},
        {"base_dir": "missing"},
    ],
)
def test_invalid_audits_are_explicit_client_errors(
    client: TestClient, payload: dict
) -> None:
    response = client.post("/api/dataset/audit", json=payload)
    assert response.status_code in (400, 422)
    assert "detail" in response.json()


def test_group_manifest_and_strict_mode_reach_the_real_audit(
    client: TestClient, tmp_path: Path
) -> None:
    for split, color in (("train", "red"), ("val", "blue")):
        folder = tmp_path / "data" / split
        folder.mkdir(parents=True)
        Image.new("RGB", (8, 8), color).save(folder / "image.png")
    (tmp_path / "data/groups.csv").write_text(
        "path,group\ntrain/image.png,plant1\nval/image.png,plant1\n"
    )
    response = client.post(
        "/api/dataset/audit",
        json={
            "base_dir": str(tmp_path / "data"),
            "splits": {"train": "train", "val": "val"},
            "group_manifest": "groups.csv",
            "strict_groups": True,
        },
    )
    assert response.status_code == 200
    report = response.json()
    assert report["ok"] is False
    assert report["duplicate_groups"] == []
    assert report["group_overlaps"][0]["group"] == "plant1"


def test_bundle_resolves_only_the_requested_profile(client: TestClient) -> None:
    profile = client.post("/api/profiles", json={"name": "Ana"}).json()
    _run(Path("outputs/models"))
    assert client.get("/api/runs/example_run/bundle").status_code == 200
    assert (
        client.get(
            "/api/runs/example_run/bundle",
            headers={
                "X-VF-Profile": profile["slug"],
            },
        ).status_code
        == 404
    )
    assert (
        client.get(
            "/api/runs/example_run/bundle",
            headers={
                "X-VF-Profile": "../other",
            },
        ).status_code
        == 400
    )
    assert (
        client.get(
            "/api/runs/example_run/bundle",
            headers={
                "X-VF-Profile": "unknown",
            },
        ).status_code
        == 404
    )


@pytest.mark.parametrize("resuming", [False, True])
@pytest.mark.parametrize("requested_id", ["example_run", "run"])
def test_active_or_resuming_runs_cannot_be_exported(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    resuming: bool,
    requested_id: str,
) -> None:
    run = _run(Path("outputs/models"))
    occupy_queue(routes, "new_resume_id" if resuming else "example_run")
    try:
        if resuming:
            assert routes._RUN_QUEUE._active is not None
            monkeypatch.setattr(routes._RUN_QUEUE._active, "resume_dir", run)
        response = client.get(f"/api/runs/{requested_id}/bundle")
        assert response.status_code == 409
    finally:
        release_queue(routes)


def test_download_temporary_files_are_cleaned_on_success_and_failure(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tempfile

    temporary = tmp_path / "temporary"
    temporary.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temporary))
    run = _run(Path("outputs/models"))
    assert client.get("/api/runs/example_run/bundle").status_code == 200
    assert list(temporary.glob("visionforge-bundle-*")) == []
    (run / "run.json").write_text("[]")
    assert client.get("/api/runs/run/bundle").status_code == 400
    assert list(temporary.glob("visionforge-bundle-*")) == []


@pytest.mark.parametrize("writer_fails", [False, True])
def test_cancellation_waits_for_open_archive_before_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, writer_fails: bool
) -> None:
    run = _run(tmp_path / "models")
    exports = evidence_routes.EvidenceExports()
    profile = default_profile(tmp_path / "models")
    router = evidence_routes.build_evidence_router(
        lambda identifier, owner: run,
        lambda header: profile,
        lambda identifier, directory, owner: False,
        exports,
    )
    endpoint = next(
        route.endpoint
        for route in router.routes
        if isinstance(route, APIRoute) and "/bundle" in route.path
    )
    started, finish = Event(), Event()
    destinations: list[Path] = []

    def blocked_writer(source: Path, destination: Path) -> Path:
        destinations.append(destination)
        with destination.open("wb") as stream:
            started.set()
            assert finish.wait(10)
            stream.write(b"completed worker")
        if writer_fails:
            raise OSError("simulated disk failure")
        return destination

    monkeypatch.setattr(evidence_routes, "export_experiment_bundle", blocked_writer)

    async def scenario() -> None:
        task = asyncio.create_task(endpoint("example_run"))
        try:
            assert await asyncio.to_thread(started.wait, 5)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            assert destinations[0].exists()
            assert exports.is_exporting(run)
            finish.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not destinations[0].parent.exists()
            assert not exports.is_exporting(run)
        finally:
            finish.set()
            if not task.done():
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


def test_continuation_cannot_start_while_archive_reads_source(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _run(Path("outputs/models"))
    original = evidence_routes.export_experiment_bundle
    started, finish = Event(), Event()

    def blocked_writer(source: Path, destination: Path) -> Path:
        started.set()
        assert finish.wait(10)
        return original(source, destination)

    monkeypatch.setattr(evidence_routes, "export_experiment_bundle", blocked_writer)
    with ThreadPoolExecutor(max_workers=1) as pool:
        download = pool.submit(client.get, "/api/runs/example_run/bundle")
        try:
            assert started.wait(5)
            response = client.post("/api/runs/example_run/resume")
            assert response.status_code == 409
            assert "export" in response.json()["detail"]
            assert not routes._RUN_QUEUE.is_busy()
        finally:
            finish.set()
        assert download.result(timeout=10).status_code == 200
    assert not routes._EVIDENCE_EXPORTS.is_exporting(run)
