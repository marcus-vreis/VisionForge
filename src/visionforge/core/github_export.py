"""Commit a run's record, without weights, to a private GitHub repository (ADR-175).

A research group keeps its experiments next to its code: one folder per run,
``runs/<experiment>/<run>/``, with a README report generated from run.json
(final metrics, a table per epoch, the run's charts), the public run.json
(ADR-174: no local path) and the charts. Weights stay out of git; they go to
the Hugging Face Hub or Zenodo.

The page previews the files and the upload must match that preview. The
repository must already exist and be private: the export never creates one
and refuses a public one. All files land in one commit on the default branch
(Git Data API); an empty repository first gets the README through the contents
API, because the Git Data API needs an existing commit.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

from visionforge.core.experiment_bundle import read_run_record
from visionforge.core.hub_export import (
    PreviewMismatchError,
    PrivateRepositoryRequiredError,
)
from visionforge.core.run_report import README, report_files, run_folder

API = "https://api.github.com"
_SITE = "github.com"
_API_VERSION = "2026-03-10"
_TIMEOUT_S = 60
_OWNER = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?")
_REPO = re.compile(r"[A-Za-z0-9_.-]{1,100}")
# GitHub's answer to an empty repository on the Git Data API.
_EMPTY = {404, 409}

# One API call: method, URL, JSON body or None -> the decoded JSON answer.
Request = Callable[[str, str, dict[str, Any] | None], Any]


class GitHubExportError(RuntimeError):
    """A GitHub call failed; the message never carries the token or the answer.

    ``status`` is the HTTP status GitHub answered with, when there was one.
    """

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class InvalidRepositoryError(ValueError):
    """The repository is not an explicit ``owner/name``."""


def is_valid_github_repo(value: str) -> bool:
    """True for an explicit ``owner/name`` GitHub accepts."""
    if not isinstance(value, str) or value.count("/") != 1:
        return False
    owner, name = value.split("/")
    return (
        _OWNER.fullmatch(owner) is not None
        and _REPO.fullmatch(name) is not None
        and name not in {".", ".."}
        and not name.endswith(".git")
    )


def preview_run(run_dir: Path) -> dict[str, Any]:
    """The folder, the files and the README a commit would hold; no network."""
    folder, files = _package(run_dir)
    return {
        "folder": folder,
        "files": [{"path": path, "size": len(data)} for path, data in files.items()],
        "readme": files[README].decode("utf-8"),
        "digest": _digest(files),
    }


def export_run(
    run_dir: Path,
    repo: str,
    token: str,
    digest: str,
    *,
    request: Request | None = None,
) -> dict[str, Any]:
    """Commit the previewed files to ``repo``'s default branch.

    Raises:
        ValueError: no token, a malformed repository, or a run that cannot be
            packaged.
        PreviewMismatchError: the files differ from the confirmed preview.
        PrivateRepositoryRequiredError: the repository is not private.
        GitHubExportError: GitHub refused or could not be reached.
    """
    if not isinstance(token, str) or not token.strip():
        raise ValueError("An explicit GitHub token is required")
    if not is_valid_github_repo(repo):
        raise InvalidRepositoryError("Repository must be an explicit owner/name")
    folder, files = _package(run_dir)
    if digest != _digest(files):
        raise PreviewMismatchError("Export changed; review and confirm a new preview")
    send = request or http_request(token)
    base = f"{API}/repos/{repo}"
    info = send("GET", base, None)
    if not isinstance(info, dict) or info.get("private") is not True:
        raise PrivateRepositoryRequiredError(
            "A verified private repository is required"
        )
    branch = info.get("default_branch")
    page = info.get("html_url")
    if not isinstance(branch, str) or not branch or not _on_github(page):
        raise GitHubExportError("GitHub did not describe the repository")
    message = f"Add VisionForge run {folder}"
    commit = _commit(send, base, branch, folder, files, message)
    return {
        "url": f"{page}/tree/{quote(branch)}/{quote(folder)}",
        "commit": commit,
        "branch": branch,
        "folder": folder,
    }


def http_request(token: str) -> Request:
    """An HTTPS caller for the GitHub API that sends only this token."""

    def call(method: str, url: str, body: dict[str, Any] | None) -> Any:
        if not url.startswith(f"{API}/"):
            raise GitHubExportError("Only the GitHub API is called")
        request = urllib.request.Request(  # noqa: S310
            # The prefix check above keeps every call on https://api.github.com.
            url,
            data=None if body is None else json.dumps(body).encode(),
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/json",
                "X-GitHub-Api-Version": _API_VERSION,
                "User-Agent": "visionforge",
            },
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:  # noqa: S310
                # Same api.github.com URL as above.
                return json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as exc:
            # The answer body can echo the request; only the status leaves.
            raise GitHubExportError(f"GitHub answered {exc.code}", exc.code) from None
        except (OSError, ValueError) as exc:
            # OSError covers URLError: no route, refused, timed out.
            raise GitHubExportError(type(exc).__name__) from None

    return call


def _commit(
    send: Request,
    base: str,
    branch: str,
    folder: str,
    files: dict[str, bytes],
    message: str,
) -> str:
    """One commit with ``files`` under ``folder`` on ``branch``; its SHA."""
    ref = f"{base}/git/ref/heads/{quote(branch)}"
    try:
        head = _sha(send("GET", ref, None), "object")
    except GitHubExportError as exc:
        if exc.status not in _EMPTY:
            raise
        # An empty repository: the contents API makes the first commit.
        readme = f"{folder}/{README}"
        send(
            "PUT",
            f"{base}/contents/{quote(readme)}",
            {"message": message, "content": _base64(files[README])},
        )
        head = _sha(send("GET", ref, None), "object")
    tree = _sha(send("GET", f"{base}/git/commits/{head}", None), "tree")
    entries = [
        {
            "path": f"{folder}/{path}",
            "mode": "100644",
            "type": "blob",
            "sha": _sha(
                send(
                    "POST",
                    f"{base}/git/blobs",
                    {"content": _base64(data), "encoding": "base64"},
                )
            ),
        }
        for path, data in files.items()
    ]
    new_tree = _sha(
        send("POST", f"{base}/git/trees", {"base_tree": tree, "tree": entries})
    )
    commit = _sha(
        send(
            "POST",
            f"{base}/git/commits",
            {"message": message, "tree": new_tree, "parents": [head]},
        )
    )
    send(
        "PATCH",
        f"{base}/git/refs/heads/{quote(branch)}",
        {"sha": commit, "force": False},
    )
    return commit


def _sha(answer: Any, key: str | None = None) -> str:
    """The ``sha`` of an answer (or of its ``key`` object); refuse anything else."""
    value = answer.get(key) if key and isinstance(answer, dict) else answer
    sha = value.get("sha") if isinstance(value, dict) else None
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40,64}", sha):
        raise GitHubExportError("GitHub returned an unexpected answer")
    return sha


def _on_github(url: Any) -> bool:
    if not isinstance(url, str):
        return False
    parts = urlsplit(url)
    return parts.scheme == "https" and parts.netloc == _SITE


def _base64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _digest(files: dict[str, bytes]) -> str:
    listing = {path: hashlib.sha256(data).hexdigest() for path, data in files.items()}
    return hashlib.sha256(json.dumps(listing, sort_keys=True).encode()).hexdigest()


def _package(run_dir: Path) -> tuple[str, dict[str, bytes]]:
    """The run's folder in the repository and its files, README first."""
    root, data = read_run_record(run_dir)
    return run_folder(root, data), report_files(root, data)
