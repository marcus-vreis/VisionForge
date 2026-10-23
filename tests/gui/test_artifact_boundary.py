"""Artifact requests authorize paths before inspecting untrusted filesystem entries."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from visionforge.gui.api.routes import router


@pytest.fixture
def artifact_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    """Serve artifacts from an isolated output root."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        yield client


@pytest.mark.parametrize("exists", [False, True])
def test_external_path_is_rejected_before_resolution(
    artifact_client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exists: bool,
) -> None:
    """External requests never reach filesystem path resolution."""
    outside = tmp_path / "private.txt"
    if exists:
        outside.write_bytes(b"private")
    original_resolve = Path.resolve

    def guarded_resolve(path: Path, strict: bool = False) -> Path:
        if path == outside:
            raise AssertionError("External path reached filesystem resolution")
        return original_resolve(path, strict=strict)

    monkeypatch.setattr(Path, "resolve", guarded_resolve)
    response = artifact_client.get(f"/api/artifacts/{outside.as_posix()}")
    assert response.status_code == 403


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("outputs/public.txt", 200),
        ("outputs/missing.txt", 404),
        ("private.txt", 403),
        ("absent.txt", 403),
        ("outputs/%2E%2E/private.txt", 403),
    ],
)
def test_artifact_containment(
    artifact_client: TestClient, tmp_path: Path, path: str, expected: int
) -> None:
    """Internal files are served and external or traversing requests are denied."""
    (tmp_path / "outputs" / "public.txt").write_bytes(b"public")
    (tmp_path / "private.txt").write_bytes(b"private")
    response = artifact_client.get(f"/api/artifacts/{path}")
    assert response.status_code == expected
    if expected == 200:
        assert response.content == b"public"


def _link_directory(link: Path, target: Path) -> None:
    if sys.platform == "win32":
        import _winapi

        _winapi.CreateJunction(str(target), str(link))
    else:
        link.symlink_to(target, target_is_directory=True)


@pytest.mark.parametrize("points_outside", [False, True])
def test_artifact_directory_aliases_cannot_cross_boundary(
    artifact_client: TestClient, tmp_path: Path, points_outside: bool
) -> None:
    """Both inward external aliases and outward internal aliases are rejected."""
    inside = tmp_path / "outputs" / "actual"
    outside = tmp_path / "private"
    inside.mkdir()
    outside.mkdir()
    (inside / "data.txt").write_bytes(b"public")
    (outside / "data.txt").write_bytes(b"private")
    link = tmp_path / "outputs" / "alias" if points_outside else tmp_path / "alias"
    _link_directory(link, outside if points_outside else inside)
    response = artifact_client.get(f"/api/artifacts/{link.as_posix()}/data.txt")
    assert response.status_code == 403
