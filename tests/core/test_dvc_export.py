"""A run is versioned with DVC in a local git repository of experiments (ADR-176).

git is the real one; DVC is ``FakeDvc``, which writes what ``dvc init`` and
``dvc add`` write so git sees the same files.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from visionforge.core import dvc_export
from visionforge.core.dvc_export import (
    DvcExportError,
    InvalidRepositoryPathError,
    RepositoryNotReadyError,
    ToolMissingError,
    Tools,
    checked_repository,
    export_run,
    find_tools,
    prepare_repository,
    preview_run,
    run_command,
)
from visionforge.core.hub_export import PreviewMismatchError

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")

_DVC = "fake-dvc"
_FOLDER = "runs/gatos-resnet18/20261108_101010"
_WEIGHTS = b"WEIGHT-BYTES-7f3a"


class FakeDvc:
    """Answers ``init``, ``add``, ``remote default`` and ``push`` like DVC 3.

    Every other command goes to the real tool (git).
    """

    def __init__(self, *, remote: str | None = None, push_ok: bool = True) -> None:
        self.remote = remote
        self.push_ok = push_ok
        self.add_ok = True
        self.calls: list[list[str]] = []

    def __call__(
        self, args: list[str], cwd: Path, timeout: float
    ) -> subprocess.CompletedProcess[str]:
        if args[0] != _DVC:
            return run_command(args, cwd, timeout)
        command = args[1:]
        self.calls.append(command)
        if command[:1] == ["init"]:
            (cwd / ".dvc").mkdir()
            (cwd / ".dvc" / "config").write_text("", encoding="utf-8")
            (cwd / ".dvc" / ".gitignore").write_text("/cache\n", encoding="utf-8")
            (cwd / ".dvcignore").write_text("", encoding="utf-8")
            run_command(["git", "add", ".dvc", ".dvcignore"], cwd, timeout)
            return _done(0)
        if command[:1] == ["add"]:
            return self._add(cwd / command[-1]) if self.add_ok else _done(1)
        if command == ["remote", "default"]:
            return _done(0, f"{self.remote}\n") if self.remote else _done(1)
        if command[:1] == ["push"]:
            return _done(0 if self.push_ok else 1)
        raise AssertionError(f"unexpected dvc call {command}")

    @staticmethod
    def _add(file: Path) -> subprocess.CompletedProcess[str]:
        data = file.read_bytes()
        md5 = hashlib.md5(data, usedforsecurity=False).hexdigest()
        pointer = f"outs:\n- md5: {md5}\n  size: {len(data)}\n  hash: md5\n  path: {file.name}\n"
        file.with_name(file.name + ".dvc").write_text(pointer, encoding="utf-8")
        ignore = file.parent / ".gitignore"
        lines = ignore.read_text(encoding="utf-8") if ignore.exists() else ""
        if f"/{file.name}\n" not in lines:
            ignore.write_text(lines + f"/{file.name}\n", encoding="utf-8")
        return _done(0)


def _done(code: int, out: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], code, out, "")


@pytest.fixture(autouse=True)
def _hermetic_git(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No global or system git config (signing, hooks path), a fixed identity."""
    empty = tmp_path / "gitconfig"
    empty.write_text("", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(empty))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for role in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{role}_NAME", "Ana")
        monkeypatch.setenv(f"GIT_{role}_EMAIL", "ana@example.org")


def _tools(fake: FakeDvc) -> Tools:
    return Tools(git=["git"], dvc=[_DVC], run=fake)


def _git(repo: Path, *args: str) -> str:
    done = run_command(["git", *args], repo, 60)
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


def _run(tmp_path: Path, *, model: str | None = "best_model.pth") -> Path:
    run_dir = tmp_path / "models" / "20261108_101010"
    run_dir.mkdir(parents=True)
    (run_dir / "best_model.pth").write_bytes(_WEIGHTS)
    (run_dir / "loss.png").write_bytes(b"\x89PNG loss")
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
            "model": str(run_dir / model) if model else None,
            "graphics": [str(run_dir / "loss.png")],
        },
    }
    (run_dir / "run.json").write_text(json.dumps(record), encoding="utf-8")
    return run_dir


def _ready_repo(tmp_path: Path, fake: FakeDvc) -> Path:
    repo = tmp_path / "experimentos"
    prepare_repository(str(repo), _tools(fake))
    return repo


def test_the_preview_puts_the_weights_in_dvc_and_the_rest_in_git(
    tmp_path: Path,
) -> None:
    fake = FakeDvc()
    preview = preview_run(
        _run(tmp_path), str(_ready_repo(tmp_path, fake)), _tools(fake)
    )

    assert preview["folder"] == _FOLDER
    storage = {item["path"]: item["storage"] for item in preview["files"]}
    assert storage == {
        "README.md": "git",
        "run.json": "git",
        "figures/loss.png": "git",
        "params.yaml": "git",
        "metrics.json": "git",
        "history.csv": "git",
        "dvc.yaml": "git",
        "model.pth": "dvc",
    }
    sizes = {item["path"]: item["size"] for item in preview["files"]}
    assert sizes["model.pth"] == len(_WEIGHTS)
    assert "`model.pth`: the weights, versioned with DVC" in preview["readme"]
    assert preview["repository"]["state"] == "ready"
    assert preview["repository"]["remote"] is None
    # The preview writes nothing into the repository.
    assert not (tmp_path / "experimentos" / "runs").exists()


def test_the_files_dvc_reads_compare_runs_without_local_paths(tmp_path: Path) -> None:
    fake = FakeDvc()
    repo = _ready_repo(tmp_path, fake)
    run_dir = _run(tmp_path)
    digest = preview_run(run_dir, str(repo), _tools(fake))["digest"]

    export_run(run_dir, str(repo), digest, _tools(fake))

    folder = repo / _FOLDER
    assert yaml.safe_load((folder / "dvc.yaml").read_text(encoding="utf-8")) == {
        "params": ["params.yaml"],
        "metrics": ["metrics.json"],
        "plots": [{"history.csv": {"x": "epoch"}}],
    }
    params = yaml.safe_load((folder / "params.yaml").read_text(encoding="utf-8"))
    assert params == {
        "task": "binary",
        "model": {"name": "resnet18"},
        "data": {"base_dir": "gatos"},
    }
    metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    assert metrics == {"accuracy": 0.9, "loss": 0.25, "total_epochs": 2.0}
    assert (folder / "history.csv").read_bytes() == (
        b"epoch,train_loss,val_accuracy\n1,0.5,0.8\n2,0.3,0.9\n"
    )
    record = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    assert record["artifacts"] == {
        "model": "model.pth",
        "graphics": ["figures/loss.png"],
        "report": "README.md",
    }
    assert str(tmp_path) not in (folder / "run.json").read_text(encoding="utf-8")


def test_prepare_makes_a_missing_folder_a_committed_dvc_repository(
    tmp_path: Path,
) -> None:
    fake = FakeDvc()
    repo = tmp_path / "experimentos"

    state = prepare_repository(str(repo), _tools(fake))

    assert state["state"] == "ready"
    assert fake.calls == [["init", "-q"], ["remote", "default"]]
    assert _git(repo, "log", "--format=%s") == "Initialize DVC for VisionForge runs"
    assert _git(repo, "status", "--porcelain") == ""
    assert _git(repo, "ls-files").splitlines() == [
        ".dvc/.gitignore",
        ".dvc/config",
        ".dvcignore",
    ]


def test_prepare_adds_dvc_to_an_existing_git_repository(tmp_path: Path) -> None:
    fake = FakeDvc()
    repo = tmp_path / "experimentos"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "notes.md").write_text("mine\n", encoding="utf-8")
    _git(repo, "add", "notes.md")

    prepare_repository(str(repo), _tools(fake))

    # Only DVC's own files went into the commit; the staged note stays staged.
    assert "notes.md" not in _git(repo, "show", "--name-only", "--format=", "HEAD")
    assert _git(repo, "status", "--porcelain") == "A  notes.md"


def test_prepare_leaves_a_ready_repository_alone(tmp_path: Path) -> None:
    fake = FakeDvc()
    repo = _ready_repo(tmp_path, fake)
    fake.calls.clear()

    prepare_repository(str(repo), _tools(fake))

    assert fake.calls == [["remote", "default"]]
    assert _git(repo, "rev-list", "--count", "HEAD") == "1"


def test_prepare_refuses_a_file_and_a_missing_parent(tmp_path: Path) -> None:
    file = tmp_path / "file.txt"
    file.write_text("x", encoding="utf-8")
    tools = _tools(FakeDvc())
    nested = str(tmp_path / "no" / "such")

    with pytest.raises(RepositoryNotReadyError):
        prepare_repository(str(file), tools)
    with pytest.raises(RepositoryNotReadyError):
        prepare_repository(nested, tools)
    assert not (tmp_path / "no").exists()


def test_the_export_commits_only_the_run_folder(tmp_path: Path) -> None:
    fake = FakeDvc()
    repo = _ready_repo(tmp_path, fake)
    (repo / "draft.md").write_text("staged by the researcher\n", encoding="utf-8")
    _git(repo, "add", "draft.md")
    run_dir = _run(tmp_path)
    digest = preview_run(run_dir, str(repo), _tools(fake))["digest"]

    result = export_run(run_dir, str(repo), digest, _tools(fake))

    assert result["commit"] == _git(repo, "rev-parse", "HEAD")
    assert result["folder"] == _FOLDER
    assert Path(result["path"]) == (repo / _FOLDER).resolve()
    committed = _git(repo, "show", "--name-only", "--format=", "HEAD").splitlines()
    assert sorted(committed) == sorted(
        f"{_FOLDER}/{name}"
        for name in (
            ".gitignore",
            "README.md",
            "dvc.yaml",
            "figures/loss.png",
            "history.csv",
            "metrics.json",
            "model.pth.dvc",
            "params.yaml",
            "run.json",
        )
    )
    # The weights are in the folder for DVC, ignored by git.
    assert (repo / _FOLDER / "model.pth").read_bytes() == _WEIGHTS
    assert (repo / _FOLDER / ".gitignore").read_text(encoding="utf-8") == "/model.pth\n"
    assert _git(repo, "status", "--porcelain") == "A  draft.md"
    assert ["add", "-q", f"{_FOLDER}/model.pth"] in fake.calls


def test_exporting_the_same_run_again_makes_no_commit(tmp_path: Path) -> None:
    fake = FakeDvc()
    repo = _ready_repo(tmp_path, fake)
    run_dir = _run(tmp_path)
    digest = preview_run(run_dir, str(repo), _tools(fake))["digest"]
    export_run(run_dir, str(repo), digest, _tools(fake))

    again = export_run(run_dir, str(repo), digest, _tools(fake))

    assert again["commit"] is None
    assert _git(repo, "rev-list", "--count", "HEAD") == "2"


def test_the_weights_are_pushed_to_the_default_remote(tmp_path: Path) -> None:
    fake = FakeDvc(remote="storage")
    repo = _ready_repo(tmp_path, fake)
    run_dir = _run(tmp_path)
    digest = preview_run(run_dir, str(repo), _tools(fake))["digest"]

    result = export_run(run_dir, str(repo), digest, _tools(fake))

    assert result["remote"] == "storage"
    assert result["pushed"] is True
    assert fake.calls[-1] == ["push", "-q", f"{_FOLDER}/model.pth.dvc"]


def test_a_failed_push_keeps_the_commit_and_says_so(tmp_path: Path) -> None:
    fake = FakeDvc(remote="storage", push_ok=False)
    repo = _ready_repo(tmp_path, fake)
    run_dir = _run(tmp_path)
    digest = preview_run(run_dir, str(repo), _tools(fake))["digest"]

    result = export_run(run_dir, str(repo), digest, _tools(fake))

    assert result["pushed"] is False
    assert result["commit"] == _git(repo, "rev-parse", "HEAD")


def test_without_a_remote_nothing_is_pushed(tmp_path: Path) -> None:
    fake = FakeDvc()
    repo = _ready_repo(tmp_path, fake)
    run_dir = _run(tmp_path)
    digest = preview_run(run_dir, str(repo), _tools(fake))["digest"]

    result = export_run(run_dir, str(repo), digest, _tools(fake))

    assert (result["remote"], result["pushed"]) == (None, None)
    assert not any(call[0] == "push" for call in fake.calls)


def test_a_run_without_weights_goes_to_git_alone(tmp_path: Path) -> None:
    fake = FakeDvc(remote="storage")
    repo = _ready_repo(tmp_path, fake)
    run_dir = _run(tmp_path, model=None)
    preview = preview_run(run_dir, str(repo), _tools(fake))

    result = export_run(run_dir, str(repo), preview["digest"], _tools(fake))

    assert all(item["storage"] == "git" for item in preview["files"])
    assert "This run recorded no weights." in preview["readme"]
    assert result["pushed"] is None
    assert not any(call[0] in {"add", "push"} for call in fake.calls)


def test_a_changed_run_is_refused_before_anything_is_written(tmp_path: Path) -> None:
    fake = FakeDvc()
    repo = _ready_repo(tmp_path, fake)
    run_dir = _run(tmp_path)
    digest = preview_run(run_dir, str(repo), _tools(fake))["digest"]
    (run_dir / "best_model.pth").write_bytes(b"retrained")
    tools = _tools(fake)

    with pytest.raises(PreviewMismatchError):
        export_run(run_dir, str(repo), digest, tools)
    assert not (repo / "runs").exists()


def test_a_preview_for_one_repository_does_not_export_to_another(
    tmp_path: Path,
) -> None:
    fake = FakeDvc()
    first = _ready_repo(tmp_path, fake)
    second = tmp_path / "outro"
    prepare_repository(str(second), _tools(fake))
    run_dir = _run(tmp_path)
    digest = preview_run(run_dir, str(first), _tools(fake))["digest"]
    tools = _tools(fake)

    with pytest.raises(PreviewMismatchError):
        export_run(run_dir, str(second), digest, tools)


@pytest.mark.parametrize("prepare", ["missing", "not_git", "no_dvc"])
def test_a_folder_that_is_not_ready_is_refused(tmp_path: Path, prepare: str) -> None:
    repo = tmp_path / "experimentos"
    if prepare != "missing":
        repo.mkdir()
    if prepare == "no_dvc":
        _git(repo, "init", "-q")
    fake = FakeDvc()
    run_dir = _run(tmp_path)
    preview = preview_run(run_dir, str(repo), _tools(fake))
    tools = _tools(fake)

    assert preview["repository"]["state"] == prepare
    with pytest.raises(RepositoryNotReadyError):
        export_run(run_dir, str(repo), preview["digest"], tools)


@pytest.mark.parametrize(
    ("tools", "missing"),
    [
        (Tools(git=None, dvc=[_DVC]), "git"),
        (Tools(git=["git"], dvc=None), "dvc"),
    ],
)
def test_a_missing_tool_is_named(tmp_path: Path, tools: Tools, missing: str) -> None:
    repo = str(tmp_path / "experimentos")

    with pytest.raises(ToolMissingError) as caught:
        prepare_repository(repo, tools)
    assert caught.value.tool == missing


def test_another_repositorys_git_variables_do_not_leak_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A server started from a git hook inherits these, pointing at that repository.
    other = tmp_path / "other"
    other.mkdir()
    _git(other, "init", "-q")
    monkeypatch.setenv("GIT_DIR", str(other / ".git"))
    monkeypatch.setenv("GIT_INDEX_FILE", str(other / ".git" / "index"))
    monkeypatch.setenv("GIT_WORK_TREE", str(other))
    fake = FakeDvc()
    repo = _ready_repo(tmp_path, fake)
    run_dir = _run(tmp_path)
    digest = preview_run(run_dir, str(repo), _tools(fake))["digest"]

    result = export_run(run_dir, str(repo), digest, _tools(fake))

    for name in ("GIT_DIR", "GIT_INDEX_FILE", "GIT_WORK_TREE"):
        monkeypatch.delenv(name)
    assert result["commit"] == _git(repo, "rev-parse", "HEAD")
    assert _git(repo, "rev-list", "--count", "HEAD") == "2"
    assert run_command(["git", "rev-parse", "HEAD"], other, 60).returncode != 0


def test_a_linked_runs_folder_cannot_lead_outside(tmp_path: Path) -> None:
    fake = FakeDvc()
    repo = _ready_repo(tmp_path, fake)
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (repo / "runs").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("this system does not let the test create a symlink")
    run_dir = _run(tmp_path)
    digest = preview_run(run_dir, str(repo), _tools(fake))["digest"]
    tools = _tools(fake)

    with pytest.raises(ValueError, match="leave the repository"):
        export_run(run_dir, str(repo), digest, tools)
    assert list(outside.iterdir()) == []


def test_a_rejected_commit_is_a_git_failure(tmp_path: Path) -> None:
    fake = FakeDvc()
    repo = _ready_repo(tmp_path, fake)
    hook = repo / ".git" / "hooks" / "pre-commit"
    hook.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    hook.chmod(0o755)
    run_dir = _run(tmp_path)
    digest = preview_run(run_dir, str(repo), _tools(fake))["digest"]
    tools = _tools(fake)

    with pytest.raises(DvcExportError) as caught:
        export_run(run_dir, str(repo), digest, tools)
    assert (caught.value.tool, caught.value.step) == ("git", "commit")


def test_a_failed_dvc_add_commits_nothing(tmp_path: Path) -> None:
    fake = FakeDvc()
    repo = _ready_repo(tmp_path, fake)
    fake.add_ok = False
    run_dir = _run(tmp_path)
    digest = preview_run(run_dir, str(repo), _tools(fake))["digest"]
    tools = _tools(fake)

    with pytest.raises(DvcExportError) as caught:
        export_run(run_dir, str(repo), digest, tools)
    assert (caught.value.tool, caught.value.step) == ("dvc", "add")
    assert _git(repo, "rev-list", "--count", "HEAD") == "1"


def test_a_command_that_cannot_start_is_a_failure(tmp_path: Path) -> None:
    def broken(args: list[str], cwd: Path, timeout: float) -> Any:
        raise subprocess.TimeoutExpired(args, timeout)

    repo = str(tmp_path / "experimentos")
    tools = Tools(git=["git"], dvc=[_DVC], run=broken)

    with pytest.raises(DvcExportError) as caught:
        prepare_repository(repo, tools)
    assert (caught.value.tool, caught.value.step) == ("git", "init")


def test_an_unsupported_checkpoint_is_refused(tmp_path: Path) -> None:
    run_dir = _run(tmp_path, model="loss.png")
    repo = str(tmp_path / "experimentos")
    tools = _tools(FakeDvc())

    with pytest.raises(ValueError, match="Unsupported checkpoint"):
        preview_run(run_dir, repo, tools)


def test_a_run_with_nothing_to_compare_has_no_dvc_yaml(tmp_path: Path) -> None:
    run_dir = _run(tmp_path)
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    record.update(config={}, metrics={}, history=[{"note": "no numbers"}])
    (run_dir / "run.json").write_text(json.dumps(record), encoding="utf-8")

    preview = preview_run(run_dir, str(tmp_path / "x"), _tools(FakeDvc()))

    names = {item["path"] for item in preview["files"]}
    assert names == {"README.md", "run.json", "figures/loss.png", "model.pth"}


@pytest.mark.parametrize("value", ["", "   ", "relative/path", "C:", "/", "a\0b"])
def test_a_repository_must_be_an_absolute_folder(value: str) -> None:
    with pytest.raises(InvalidRepositoryPathError):
        checked_repository(value)


def test_an_absolute_folder_is_accepted(tmp_path: Path) -> None:
    assert checked_repository(f"  {tmp_path}  ") == tmp_path.resolve()


def test_dvc_comes_from_this_environment_first(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dvc_export.importlib.util, "find_spec", lambda name: object())

    assert find_tools().dvc == [sys.executable, "-m", "dvc"]


def test_dvc_comes_from_the_path_without_the_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dvc_export.importlib.util, "find_spec", lambda name: None)
    found = {"dvc": "/opt/bin/dvc", "git": "/usr/bin/git"}
    monkeypatch.setattr(dvc_export.shutil, "which", found.get)

    assert find_tools() == Tools(git=["/usr/bin/git"], dvc=["/opt/bin/dvc"])


def test_no_dvc_anywhere_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dvc_export.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(dvc_export.shutil, "which", lambda name: None)

    assert find_tools() == Tools(git=None, dvc=None)


def test_commands_run_without_a_prompt_or_dvc_analytics(tmp_path: Path) -> None:
    probe = "import os, sys; print(os.environ['DVC_NO_ANALYTICS'], sys.stdin.read())"

    done = run_command([sys.executable, "-c", probe], tmp_path, 60)

    assert done.returncode == 0
    assert done.stdout.strip() == "1"
