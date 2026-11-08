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
import math
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote, urlsplit

from visionforge.core.experiment_bundle import contained_run_file, read_run_record
from visionforge.core.hub_export import (
    PreviewMismatchError,
    PrivateRepositoryRequiredError,
)
from visionforge.core.public_run_record import without_local_paths

API = "https://api.github.com"
_SITE = "github.com"
_API_VERSION = "2026-03-10"
_TIMEOUT_S = 60
_FIGURES = {".png", ".jpg", ".jpeg"}
_MAX_FIGURE = 5 * 1024 * 1024
_MAX_FIGURES = 20
_MAX_EPOCH_ROWS = 200
_MAX_COLUMNS = 8
_SEGMENT = re.compile(r"[^A-Za-z0-9_.-]+")
_OWNER = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?")
_REPO = re.compile(r"[A-Za-z0-9_.-]{1,100}")
_README = "README.md"
_RECORD = "run.json"
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
        "readme": files[_README].decode("utf-8"),
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
        raise ValueError("Repository must be an explicit owner/name")
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
        readme = f"{folder}/{_README}"
        send(
            "PUT",
            f"{base}/contents/{quote(readme)}",
            {"message": message, "content": _base64(files[_README])},
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
    experiment = data.get("experiment")
    folder = "runs/" + "/".join(
        _segment(name) for name in (experiment, root.name) if _segment(name)
    )
    figures = _figures(root, data.get("artifacts", {}).get("graphics"))
    record = without_local_paths(
        {key: value for key, value in data.items() if key != "artifacts"}
    )
    record["run_dir"] = "."
    record["artifacts"] = {"model": None, "graphics": list(figures), "report": _README}
    files = {
        _README: _readme(data, folder, list(figures)).encode("utf-8"),
        _RECORD: json.dumps(record, indent=2, ensure_ascii=False).encode("utf-8"),
        **figures,
    }
    return folder, files


def _segment(name: Any) -> str:
    """A path segment made of safe characters, or '' for nothing usable."""
    if not isinstance(name, str):
        return ""
    return _SEGMENT.sub("-", name).strip("-.")[:100]


def _figures(root: Path, graphics: Any) -> dict[str, bytes]:
    """The run's charts as ``figures/<name>``: images only, bounded in size."""
    figures: dict[str, bytes] = {}
    values = graphics if isinstance(graphics, list) else []
    for value in values[:_MAX_FIGURES]:
        if not isinstance(value, str):
            continue
        try:
            path = contained_run_file(root, value)
        except (OSError, ValueError):
            continue
        if path.suffix.lower() not in _FIGURES or path.stat().st_size > _MAX_FIGURE:
            continue
        figures[f"figures/{_segment(path.name)}"] = path.read_bytes()
    return figures


def _readme(data: dict[str, Any], folder: str, figures: list[str]) -> str:
    """A Markdown report of the run, from run.json alone."""
    lines = [f"# {PurePosixPath(folder).name}", "", *_summary(data), ""]
    lines += ["## Final metrics", "", *_table([_numbers(data.get("metrics"))])]
    lines += _epochs(data.get("history"))
    if figures:
        lines += ["## Charts", ""]
        lines += [f"![{PurePosixPath(path).stem}]({path})" for path in figures]
        lines.append("")
    lines += [
        "## Files",
        "",
        "- `run.json`: the run record, with this computer's paths reduced to their "
        "final name.",
        "- The weights are not in this repository.",
        "",
        "Exported by VisionForge.",
        "",
    ]
    return "\n".join(lines)


def _summary(data: dict[str, Any]) -> list[str]:
    """``- **Label:** value`` lines for the run's plain descriptive fields."""
    config = _mapping(data.get("config"))
    fields = (
        ("Experiment", data.get("experiment")),
        ("Task", config.get("task")),
        ("Model", _mapping(config.get("model")).get("name")),
        ("Status", data.get("status")),
        ("Date", data.get("timestamp")),
    )
    return [
        f"- **{label}:** {_cell(value)}"
        for label, value in fields
        if isinstance(value, str) and value
    ]


def _epochs(history: Any) -> list[str]:
    """The per-epoch table, bounded; nothing when no epoch has numbers."""
    epochs = history if isinstance(history, list) else []
    rows = [_numbers(row) for row in epochs[:_MAX_EPOCH_ROWS]]
    if not any(rows):
        return []
    lines = ["## Per epoch", "", *_table(rows)]
    if len(epochs) > _MAX_EPOCH_ROWS:
        lines += [f"First {_MAX_EPOCH_ROWS} of {len(epochs)} epochs.", ""]
    return lines


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _numbers(row: Any) -> dict[str, float]:
    """The finite numeric values of a mapping, keyed by plain names."""
    if not isinstance(row, dict):
        return {}
    return {
        str(key): float(value)
        for key, value in row.items()
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and re.fullmatch(r"[A-Za-z0-9_./-]{1,60}", str(key))
    }


def _table(rows: list[dict[str, float]]) -> list[str]:
    """A Markdown table of the first ``_MAX_COLUMNS`` names; nothing if empty."""
    columns = list(dict.fromkeys(key for row in rows for key in row))[:_MAX_COLUMNS]
    if not columns:
        return ["No numeric values were recorded.", ""]
    header = "| " + " | ".join(columns) + " |"
    rule = "|" + "---|" * len(columns)
    body = [
        "| " + " | ".join(_number(row.get(key)) for key in columns) + " |"
        for row in rows
    ]
    return [header, rule, *body, ""]


def _number(value: float | None) -> str:
    if value is None:
        return ""
    return (
        str(int(value)) if value.is_integer() and abs(value) < 1e15 else f"{value:.4g}"
    )


def _cell(text: str) -> str:
    """Text safe inside one Markdown line: no newline, pipe or HTML."""
    return re.sub(r"[\r\n|<>`]+", " ", text)[:120]
