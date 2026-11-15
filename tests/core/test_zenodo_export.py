"""A run goes to Zenodo as an unpublished draft holding its public bundle."""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
import zipfile
from collections.abc import Mapping
from email.message import Message
from pathlib import Path
from typing import Any, BinaryIO

import pytest

from visionforge.core import zenodo_export
from visionforge.core.hub_export import PreviewMismatchError
from visionforge.core.zenodo_export import (
    PRODUCTION,
    SANDBOX,
    ZenodoExportError,
    export_run,
    http_request,
    preview_run,
    zenodo_link,
)

_TOKEN = "zenodo-test-token-123456"


class FakeZenodo:
    """Answers the deposit API like Zenodo does and records every call."""

    def __init__(self, base: str, bucket_host: str | None = None) -> None:
        host = bucket_host or base
        self.calls: list[tuple[str, str]] = []
        self.uploaded = b""
        self.deposition = {
            "id": 1234,
            "links": {
                "bucket": f"{host}/api/files/bucket-uuid",
                "html": f"{base}/deposit/1234",
            },
        }

    def __call__(
        self,
        method: str,
        url: str,
        body: bytes | BinaryIO,
        headers: Mapping[str, str],
    ) -> Any:
        self.calls.append((method, url))
        if method == "PUT":
            assert not isinstance(body, bytes)
            self.uploaded = body.read()
            assert headers["Content-Length"] == str(len(self.uploaded))
            return {"key": url.rsplit("/", 1)[-1]}
        return self.deposition


def _run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "models" / "gatos_001"
    run_dir.mkdir(parents=True)
    (run_dir / "best_model.pth").write_bytes(b"weights")
    record = {
        "run_dir": str(run_dir),
        "config": {"data": {"base_dir": "/home/ana/datasets/gatos"}},
        "artifacts": {"model": str(run_dir / "best_model.pth")},
        "metrics": {"accuracy": 0.9},
    }
    (run_dir / "run.json").write_text(json.dumps(record), encoding="utf-8")
    return run_dir


def test_the_preview_lists_the_bundle_and_is_stable(tmp_path: Path) -> None:
    run_dir = _run(tmp_path)

    first = preview_run(run_dir)

    assert first["name"] == "visionforge-gatos_001.zip"
    assert first["files"] == ["best_model.pth", "manifest.json", "run.json"]
    assert {
        "artifact": "run.json",
        "reason": "local paths reduced to their final name",
    } in first["notes"]
    assert first["size"] > 0
    assert preview_run(run_dir)["digest"] == first["digest"]


@pytest.mark.parametrize(
    ("experiment", "name"),
    [
        ("gatos resnet18", "visionforge-gatos-resnet18-gatos_001.zip"),
        ("gatos_001", "visionforge-gatos_001.zip"),
        ("../..", "visionforge-gatos_001.zip"),
        (7, "visionforge-gatos_001.zip"),
    ],
)
def test_the_bundle_is_named_after_its_experiment(
    tmp_path: Path, experiment: object, name: str
) -> None:
    run_dir = _run(tmp_path)
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    record["experiment"] = experiment
    (run_dir / "run.json").write_text(json.dumps(record), encoding="utf-8")

    assert preview_run(run_dir)["name"] == name


def test_a_draft_is_created_and_the_bundle_uploaded_never_published(
    tmp_path: Path,
) -> None:
    run_dir = _run(tmp_path)
    digest = preview_run(run_dir)["digest"]
    zenodo = FakeZenodo(PRODUCTION)

    result = export_run(run_dir, _TOKEN, digest, request=zenodo)

    assert zenodo.calls == [
        ("POST", f"{PRODUCTION}/api/deposit/depositions"),
        ("PUT", f"{PRODUCTION}/api/files/bucket-uuid/visionforge-gatos_001.zip"),
    ]
    assert result == {
        "id": 1234,
        "url": f"{PRODUCTION}/deposit/1234",
        "sandbox": False,
        "name": "visionforge-gatos_001.zip",
    }
    with zipfile.ZipFile(io.BytesIO(zenodo.uploaded)) as bundle:
        record = json.loads(bundle.read("run.json"))
        assert bundle.read("best_model.pth") == b"weights"
    assert record["config"]["data"]["base_dir"] == "gatos"
    assert record["run_dir"] == "."


def test_the_sandbox_is_a_separate_site(tmp_path: Path) -> None:
    run_dir = _run(tmp_path)
    zenodo = FakeZenodo(SANDBOX)

    result = export_run(
        run_dir, _TOKEN, preview_run(run_dir)["digest"], sandbox=True, request=zenodo
    )

    assert all(url.startswith(f"{SANDBOX}/") for _, url in zenodo.calls)
    assert result["sandbox"] is True
    assert result["url"] == f"{SANDBOX}/deposit/1234"


def test_a_changed_run_needs_a_new_preview(tmp_path: Path) -> None:
    run_dir = _run(tmp_path)
    digest = preview_run(run_dir)["digest"]
    (run_dir / "best_model.pth").write_bytes(b"other weights")
    zenodo = FakeZenodo(PRODUCTION)

    with pytest.raises(PreviewMismatchError):
        export_run(run_dir, _TOKEN, digest, request=zenodo)
    assert zenodo.calls == []


def test_the_token_never_follows_a_bucket_outside_zenodo(tmp_path: Path) -> None:
    run_dir = _run(tmp_path)
    zenodo = FakeZenodo(PRODUCTION, bucket_host="https://files.example.org")

    digest = preview_run(run_dir)["digest"]

    with pytest.raises(ZenodoExportError, match="outside Zenodo"):
        export_run(run_dir, _TOKEN, digest, request=zenodo)
    assert [method for method, _ in zenodo.calls] == ["POST"]


@pytest.mark.parametrize(
    "links",
    [
        None,
        {},
        {"bucket": 3},
        {"bucket": "http://zenodo.org/api/files/x"},
        {"bucket": "https://zenodo.org.example.org/api/files/x"},
    ],
)
def test_a_link_must_be_https_on_the_same_host(links: Any) -> None:
    with pytest.raises(ZenodoExportError):
        zenodo_link(links, "bucket", PRODUCTION)


def test_an_empty_token_is_refused_before_any_call(tmp_path: Path) -> None:
    zenodo = FakeZenodo(PRODUCTION)
    run_dir = _run(tmp_path)
    with pytest.raises(ValueError, match="token"):
        export_run(run_dir, " ", "digest", request=zenodo)
    assert zenodo.calls == []


class _Answer(io.BytesIO):
    def __enter__(self) -> _Answer:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def test_the_http_caller_sends_the_token_as_a_bearer_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[urllib.request.Request] = []

    def answer(request: urllib.request.Request, timeout: float) -> _Answer:
        seen.append(request)
        return _Answer(b'{"id": 7}')

    monkeypatch.setattr(zenodo_export.urllib.request, "urlopen", answer)
    send = http_request(_TOKEN)

    assert send("POST", f"{PRODUCTION}/api/deposit/depositions", b"{}", {}) == {"id": 7}
    assert seen[0].get_header("Authorization") == f"Bearer {_TOKEN}"
    assert _TOKEN not in seen[0].full_url


def test_a_refusal_keeps_the_status_and_drops_the_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(request: urllib.request.Request, timeout: float) -> _Answer:
        raise urllib.error.HTTPError(
            request.full_url,
            403,
            f"echo {_TOKEN}",
            Message(),
            io.BytesIO(_TOKEN.encode()),
        )

    monkeypatch.setattr(zenodo_export.urllib.request, "urlopen", refuse)

    send = http_request(_TOKEN)

    with pytest.raises(ZenodoExportError) as caught:
        send("POST", f"{PRODUCTION}/api", b"{}", {})
    assert caught.value.status == 403
    assert _TOKEN not in str(caught.value)
    assert caught.value.__cause__ is None


def test_the_http_caller_refuses_plain_http() -> None:
    send = http_request(_TOKEN)
    with pytest.raises(ZenodoExportError, match="HTTPS"):
        send("POST", "http://zenodo.org/api", b"{}", {})


def test_a_network_failure_names_only_its_kind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(request: urllib.request.Request, timeout: float) -> _Answer:
        raise urllib.error.URLError(f"no route {_TOKEN}")

    monkeypatch.setattr(zenodo_export.urllib.request, "urlopen", fail)

    send = http_request(_TOKEN)

    with pytest.raises(ZenodoExportError, match="^URLError$") as caught:
        send("POST", f"{PRODUCTION}/api", b"{}", {})
    assert caught.value.status is None
