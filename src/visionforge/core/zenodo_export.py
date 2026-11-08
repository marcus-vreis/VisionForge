"""Send a run's experiment bundle to Zenodo as an unpublished draft (ADR-174).

Publishing on Zenodo mints a permanent DOI and cannot be taken back, so the
export stops at the draft: it creates an empty deposition and puts the bundle
in its file bucket. Title, authors, license and the decision to publish stay
with the researcher on Zenodo. The bundle is the ADR-135 one with a public
run.json (no local paths); the page previews it and the upload must match the
previewed manifest.
"""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import quote, urlsplit

from visionforge.core.experiment_bundle import (
    MANIFEST_FILENAME,
    RUN_FILENAME,
    export_experiment_bundle,
)
from visionforge.core.hub_export import PreviewMismatchError

PRODUCTION = "https://zenodo.org"
SANDBOX = "https://sandbox.zenodo.org"
_TIMEOUT_S = 120
_STEM = re.compile(r"[^A-Za-z0-9_.-]+")
_MAX_STEM = 150

# One HTTP call: method, URL, body, extra headers -> the decoded JSON answer.
Request = Callable[[str, str, bytes | BinaryIO, Mapping[str, str]], Any]


class ZenodoExportError(RuntimeError):
    """A Zenodo call failed; the message never carries the token or the answer.

    ``status`` is the HTTP status Zenodo answered with, when there was one.
    """

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def preview_run(run_dir: Path) -> dict[str, Any]:
    """Describe the exact bundle a draft would receive, without the network."""
    with tempfile.TemporaryDirectory(prefix="visionforge-zenodo-") as folder:
        return _describe(_build(run_dir, Path(folder)))


def export_run(
    run_dir: Path,
    token: str,
    digest: str,
    *,
    sandbox: bool = False,
    request: Request | None = None,
) -> dict[str, Any]:
    """Create a draft deposition holding the previewed bundle; never publish.

    Raises:
        ValueError: no token, or the run cannot be bundled.
        PreviewMismatchError: the bundle differs from the confirmed preview.
        ZenodoExportError: Zenodo refused or could not be reached.
    """
    if not isinstance(token, str) or not token.strip():
        raise ValueError("An explicit Zenodo token is required")
    base = SANDBOX if sandbox else PRODUCTION
    send = request or http_request(token)
    with tempfile.TemporaryDirectory(prefix="visionforge-zenodo-") as folder:
        bundle = _build(run_dir, Path(folder))
        described = _describe(bundle)
        if digest != described["digest"]:
            raise PreviewMismatchError(
                "Export changed; review and confirm a new preview"
            )
        deposition = send(
            "POST",
            f"{base}/api/deposit/depositions",
            b"{}",
            {"Content-Type": "application/json"},
        )
        links = deposition.get("links") if isinstance(deposition, dict) else None
        bucket = zenodo_link(links, "bucket", base)
        page = zenodo_link(links, "html", base)
        with bundle.open("rb") as stream:
            send(
                "PUT",
                f"{bucket}/{quote(bundle.name)}",
                stream,
                {
                    "Content-Type": "application/zip",
                    "Content-Length": str(described["size"]),
                },
            )
    identifier = deposition.get("id")
    return {
        "id": identifier if isinstance(identifier, int) else None,
        "url": page,
        "sandbox": sandbox,
        "name": described["name"],
    }


def zenodo_link(links: Any, key: str, base: str) -> str:
    """``links[key]`` when it is an https URL on ``base``'s host; else refuse.

    The bucket link receives the token, so it must stay on Zenodo.
    """
    value = links.get(key) if isinstance(links, dict) else None
    if not isinstance(value, str):
        raise ZenodoExportError("Zenodo did not return the draft links")
    parts = urlsplit(value)
    if parts.scheme != "https" or parts.netloc != urlsplit(base).netloc:
        raise ZenodoExportError("Zenodo returned a link outside Zenodo")
    return value.rstrip("/")


def http_request(token: str) -> Request:
    """An HTTPS caller that sends only this token, as a Bearer header."""

    def call(
        method: str, url: str, body: bytes | BinaryIO, headers: Mapping[str, str]
    ) -> Any:
        if not url.startswith("https://"):
            raise ZenodoExportError("The Zenodo address must use HTTPS")
        request = urllib.request.Request(  # noqa: S310
            # The scheme is checked above: only https reaches urlopen.
            url,
            data=body,
            headers={
                **headers,
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
                "User-Agent": "visionforge",
            },
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:  # noqa: S310
                # Same https URL as above.
                return json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as exc:
            # The answer body can echo the request; only the status leaves.
            raise ZenodoExportError(f"Zenodo answered {exc.code}", exc.code) from None
        except (OSError, ValueError) as exc:
            # OSError covers URLError: no route, refused, timed out.
            raise ZenodoExportError(type(exc).__name__) from None

    return call


def _build(run_dir: Path, folder: Path) -> Path:
    """The public bundle, named after its experiment and run folder."""
    bundle = export_experiment_bundle(run_dir, folder / "bundle.zip", public=True)
    with zipfile.ZipFile(bundle) as archive:
        record = json.loads(archive.read(RUN_FILENAME))
    experiment = record.get("experiment") if isinstance(record, dict) else None
    names = [experiment if isinstance(experiment, str) else "", run_dir.name]
    parts = dict.fromkeys(_STEM.sub("-", name).strip("-.") for name in names)
    stem = "-".join(part for part in parts if part)[:_MAX_STEM] or "run"
    return bundle.rename(folder / f"visionforge-{stem}.zip")


def _describe(bundle: Path) -> dict[str, Any]:
    with zipfile.ZipFile(bundle) as archive:
        manifest = archive.read(MANIFEST_FILENAME)
        files = sorted(archive.namelist())
    notes = json.loads(manifest).get("notes", [])
    return {
        "name": bundle.name,
        "size": bundle.stat().st_size,
        "files": files,
        "notes": [
            {"artifact": str(note.get("artifact")), "reason": str(note.get("reason"))}
            for note in notes
            if isinstance(note, dict)
        ],
        "digest": hashlib.sha256(manifest).hexdigest(),
    }
