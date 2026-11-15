"""A run's record goes to a private GitHub repository in one commit, without weights."""

from __future__ import annotations

import base64
import io
import json
import urllib.error
import urllib.request
from email.message import Message
from pathlib import Path
from typing import Any

import pytest

from visionforge.core import github_export
from visionforge.core.github_export import (
    API,
    GitHubExportError,
    export_run,
    http_request,
    is_valid_github_repo,
    preview_run,
)
from visionforge.core.hub_export import (
    PreviewMismatchError,
    PrivateRepositoryRequiredError,
)

_TOKEN = "github-test-token-123456"
_REPO = "ana/experimentos"
_BASE = f"{API}/repos/{_REPO}"
_HEAD = "a" * 40


class FakeGitHub:
    """Answers the repository and Git Data endpoints and records every call."""

    def __init__(self, *, private: bool = True, empty: bool = False) -> None:
        self.private = private
        self.empty = empty
        self.calls: list[tuple[str, str]] = []
        self.blobs: list[bytes] = []
        self.tree: list[dict[str, Any]] = []
        self.contents: dict[str, bytes] = {}
        self.ref_update: dict[str, Any] | None = None

    def __call__(self, method: str, url: str, body: dict[str, Any] | None) -> Any:
        self.calls.append((method, url))
        route = url.removeprefix(_BASE)
        if route == "":
            return {
                "private": self.private,
                "default_branch": "main",
                "html_url": f"https://github.com/{_REPO}",
            }
        if route == "/git/ref/heads/main":
            if self.empty:
                raise GitHubExportError("GitHub answered 409", 409)
            return {"object": {"sha": _HEAD}}
        if route.startswith("/contents/"):
            assert body is not None
            self.contents[route.removeprefix("/contents/")] = base64.b64decode(
                body["content"]
            )
            self.empty = False
            return {"content": {"sha": "c" * 40}}
        if route == f"/git/commits/{_HEAD}":
            return {"tree": {"sha": "b" * 40}}
        assert body is not None
        if route == "/git/blobs":
            self.blobs.append(base64.b64decode(body["content"]))
            return {"sha": f"{len(self.blobs):040x}"}
        if route == "/git/trees":
            assert body["base_tree"] == "b" * 40
            self.tree = body["tree"]
            return {"sha": "d" * 40}
        if route == "/git/commits":
            assert body["parents"] == [_HEAD]
            return {"sha": "e" * 40}
        if route == "/git/refs/heads/main":
            self.ref_update = body
            return {"object": {"sha": "e" * 40}}
        raise AssertionError(f"unexpected call {method} {url}")


def _run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "models" / "20261108_101010"
    run_dir.mkdir(parents=True)
    (run_dir / "best_model.pth").write_bytes(b"WEIGHT-BYTES-7f3a")
    (run_dir / "loss.png").write_bytes(b"\x89PNG loss")
    (run_dir / "notes.txt").write_text("not a chart", encoding="utf-8")
    record = {
        "experiment": "gatos resnet18",
        "status": "completed",
        "timestamp": "2026-11-08T10:10:10",
        "run_dir": str(run_dir),
        "config": {
            "task": "binary",
            "model": {"name": "resnet18"},
            "data": {"base_dir": "/home/ana/datasets/gatos"},
        },
        "metrics": {"accuracy": 0.9, "loss": 0.25, "total_epochs": 2},
        "history": [
            {"epoch": 1, "train_loss": 0.5, "val_accuracy": 0.8},
            {"epoch": 2, "train_loss": 0.3, "val_accuracy": 0.9},
        ],
        "artifacts": {
            "model": str(run_dir / "best_model.pth"),
            "graphics": [str(run_dir / "loss.png"), str(run_dir / "notes.txt")],
            "report": None,
        },
    }
    (run_dir / "run.json").write_text(json.dumps(record), encoding="utf-8")
    return run_dir


@pytest.mark.parametrize(
    "repo", ["ana/experimentos", "a/b", "lab-ufx/runs.2026", "Ana-1/x_y"]
)
def test_an_owner_and_name_are_accepted(repo: str) -> None:
    assert is_valid_github_repo(repo)


@pytest.mark.parametrize(
    "repo",
    [
        "",
        "ana",
        "ana/",
        "/x",
        "ana/x/y",
        "-ana/x",
        "a_b/x",
        "ana/..",
        "ana/x.git",
        "a b/c",
        3,
    ],
)
def test_anything_else_is_refused(repo: Any) -> None:
    assert not is_valid_github_repo(repo)


def test_the_preview_has_a_report_the_record_and_the_charts_but_no_weights(
    tmp_path: Path,
) -> None:
    preview = preview_run(_run(tmp_path))

    assert preview["folder"] == "runs/gatos-resnet18/20261108_101010"
    assert [file["path"] for file in preview["files"]] == [
        "README.md",
        "run.json",
        "figures/loss.png",
    ]
    readme = preview["readme"]
    assert "- **Model:** resnet18" in readme
    assert "| accuracy | loss | total_epochs |" in readme
    assert "| 0.9 | 0.25 | 2 |" in readme
    assert "| epoch | train_loss | val_accuracy |" in readme
    assert "| 2 | 0.3 | 0.9 |" in readme
    assert "![loss](figures/loss.png)" in readme
    assert "weights are not in this repository" in readme
    # The same run elsewhere on disk is the same package.
    assert preview_run(_run(tmp_path / "again"))["digest"] == preview["digest"]


def test_one_commit_holds_every_file_under_the_run_folder(tmp_path: Path) -> None:
    run_dir = _run(tmp_path)
    digest = preview_run(run_dir)["digest"]
    github = FakeGitHub()

    result = export_run(run_dir, _REPO, _TOKEN, digest, request=github)

    folder = "runs/gatos-resnet18/20261108_101010"
    assert [entry["path"] for entry in github.tree] == [
        f"{folder}/README.md",
        f"{folder}/run.json",
        f"{folder}/figures/loss.png",
    ]
    assert github.ref_update == {"sha": "e" * 40, "force": False}
    assert result == {
        "url": f"https://github.com/{_REPO}/tree/main/{folder}",
        "commit": "e" * 40,
        "branch": "main",
        "folder": folder,
    }
    record = json.loads(github.blobs[1])
    assert record["run_dir"] == "."
    assert record["config"]["data"]["base_dir"] == "gatos"
    assert record["artifacts"] == {
        "model": None,
        "graphics": ["figures/loss.png"],
        "report": "README.md",
    }
    assert b"WEIGHT-BYTES-7f3a" not in b"".join(github.blobs)
    assert str(tmp_path) not in github.blobs[1].decode()


def test_an_empty_repository_gets_its_first_commit_from_the_readme(
    tmp_path: Path,
) -> None:
    run_dir = _run(tmp_path)
    github = FakeGitHub(empty=True)

    export_run(run_dir, _REPO, _TOKEN, preview_run(run_dir)["digest"], request=github)

    readme = "runs/gatos-resnet18/20261108_101010/README.md"
    assert github.contents[readme].startswith(b"# 20261108_101010")
    assert github.ref_update is not None


def test_a_public_repository_is_refused_before_any_write(tmp_path: Path) -> None:
    run_dir = _run(tmp_path)
    github = FakeGitHub(private=False)
    digest = preview_run(run_dir)["digest"]

    with pytest.raises(PrivateRepositoryRequiredError):
        export_run(run_dir, _REPO, _TOKEN, digest, request=github)
    assert github.calls == [("GET", _BASE)]


def test_a_changed_run_needs_a_new_preview(tmp_path: Path) -> None:
    run_dir = _run(tmp_path)
    digest = preview_run(run_dir)["digest"]
    (run_dir / "loss.png").write_bytes(b"\x89PNG new curve")
    github = FakeGitHub()

    with pytest.raises(PreviewMismatchError):
        export_run(run_dir, _REPO, _TOKEN, digest, request=github)
    assert github.calls == []


@pytest.mark.parametrize(("repo", "token"), [("ana", _TOKEN), (_REPO, " ")])
def test_a_bad_repository_or_token_is_refused_offline(
    tmp_path: Path, repo: str, token: str
) -> None:
    run_dir = _run(tmp_path)
    github = FakeGitHub()
    with pytest.raises(ValueError):
        export_run(run_dir, repo, token, "digest", request=github)
    assert github.calls == []


def test_another_ref_failure_is_not_taken_for_an_empty_repository(
    tmp_path: Path,
) -> None:
    run_dir = _run(tmp_path)
    digest = preview_run(run_dir)["digest"]
    github = FakeGitHub()

    def forbidden(method: str, url: str, body: dict[str, Any] | None) -> Any:
        if url.endswith("/git/ref/heads/main"):
            raise GitHubExportError("GitHub answered 403", 403)
        return github(method, url, body)

    with pytest.raises(GitHubExportError) as caught:
        export_run(run_dir, _REPO, _TOKEN, digest, request=forbidden)
    assert caught.value.status == 403
    assert github.contents == {}


def test_an_unexpected_answer_is_refused(tmp_path: Path) -> None:
    run_dir = _run(tmp_path)
    digest = preview_run(run_dir)["digest"]
    github = FakeGitHub()

    def odd(method: str, url: str, body: dict[str, Any] | None) -> Any:
        if url.endswith("/git/blobs"):
            return {"sha": "not-a-sha"}
        return github(method, url, body)

    with pytest.raises(GitHubExportError, match="unexpected"):
        export_run(run_dir, _REPO, _TOKEN, digest, request=odd)
    assert github.ref_update is None


def test_a_repository_page_outside_github_is_refused(tmp_path: Path) -> None:
    run_dir = _run(tmp_path)
    digest = preview_run(run_dir)["digest"]

    def elsewhere(method: str, url: str, body: dict[str, Any] | None) -> Any:
        return {
            "private": True,
            "default_branch": "main",
            "html_url": "https://x.org/a",
        }

    with pytest.raises(GitHubExportError, match="describe"):
        export_run(run_dir, _REPO, _TOKEN, digest, request=elsewhere)


def test_a_long_history_is_cut_and_said_so(tmp_path: Path) -> None:
    run_dir = _run(tmp_path)
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    record["history"] = [{"epoch": i, "loss": 1 / i} for i in range(1, 251)]
    record["metrics"] = {}
    (run_dir / "run.json").write_text(json.dumps(record), encoding="utf-8")

    readme = preview_run(run_dir)["readme"]

    assert "First 200 of 250 epochs." in readme
    assert "| 201 |" not in readme
    assert "No numeric values were recorded." in readme


def test_text_from_the_record_cannot_break_the_report(tmp_path: Path) -> None:
    run_dir = _run(tmp_path)
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    record["status"] = "done | <script>x</script>\n# injected"
    (run_dir / "run.json").write_text(json.dumps(record), encoding="utf-8")

    readme = preview_run(run_dir)["readme"]

    assert "<script>" not in readme
    assert "\n# injected" not in readme


class _Answer(io.BytesIO):
    def __enter__(self) -> _Answer:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def test_the_http_caller_sends_the_token_only_to_the_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[urllib.request.Request] = []

    def answer(request: urllib.request.Request, timeout: float) -> _Answer:
        seen.append(request)
        return _Answer(b'{"private": true}')

    monkeypatch.setattr(github_export.urllib.request, "urlopen", answer)
    send = http_request(_TOKEN)

    assert send("GET", _BASE, None) == {"private": True}
    assert seen[0].get_header("Authorization") == f"Bearer {_TOKEN}"
    assert seen[0].data is None
    with pytest.raises(GitHubExportError, match="Only the GitHub API"):
        send("GET", "https://github.com.example.org/repos/x/y", None)


def test_a_refusal_keeps_the_status_and_drops_the_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(request: urllib.request.Request, timeout: float) -> _Answer:
        raise urllib.error.HTTPError(
            request.full_url,
            401,
            f"echo {_TOKEN}",
            Message(),
            io.BytesIO(_TOKEN.encode()),
        )

    monkeypatch.setattr(github_export.urllib.request, "urlopen", refuse)
    send = http_request(_TOKEN)

    with pytest.raises(GitHubExportError) as caught:
        send("GET", _BASE, None)
    assert caught.value.status == 401
    assert _TOKEN not in str(caught.value)


def test_a_network_failure_names_only_its_kind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(request: urllib.request.Request, timeout: float) -> _Answer:
        raise urllib.error.URLError(f"no route {_TOKEN}")

    monkeypatch.setattr(github_export.urllib.request, "urlopen", fail)
    send = http_request(_TOKEN)

    with pytest.raises(GitHubExportError, match="^URLError$") as caught:
        send("POST", f"{_BASE}/git/blobs", {"content": ""})
    assert caught.value.status is None
