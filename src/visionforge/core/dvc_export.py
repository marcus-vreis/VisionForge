"""Version a run with DVC in a local repository of experiments (ADR-176).

The researcher keeps one git repository with DVC for the group's experiments,
on this computer. A run becomes the folder ``runs/<experiment>/<run>/``:

- in git: the README report and public run.json of ADR-175, the charts, and the
  files DVC reads to compare runs (``params.yaml``, ``metrics.json``,
  ``history.csv`` and a ``dvc.yaml`` that declares them, without stages), so
  ``dvc metrics show``, ``dvc params diff`` and ``dvc plots show`` work across
  runs and commits;
- in DVC: the weights, ``model.<ext>``, added with ``dvc add``; git gets the
  small ``model.<ext>.dvc`` pointer.

The page previews the files and the export must match that preview. The export
commits only that folder and runs ``dvc push`` when the repository has a default
remote; it never runs ``git push``. git and DVC are the real command-line tools
(DVC from the ``[dvc]`` extra or the PATH), run without a terminal and with
DVC's usage analytics off.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from loguru import logger

from visionforge.core.experiment_bundle import contained_run_file, read_run_record
from visionforge.core.hub_export import PreviewMismatchError
from visionforge.core.run_report import (
    README,
    numeric_values,
    public_record,
    report_files,
    run_folder,
)

DVC_FILE = "dvc.yaml"
PARAMS_FILE = "params.yaml"
METRICS_FILE = "metrics.json"
HISTORY_FILE = "history.csv"
_WEIGHTS = {".pt", ".pth", ".onnx", ".safetensors"}
_MAX_PATH = 1000
_CHUNK = 1024 * 1024
_GIT_TIMEOUT_S = 120
_ADD_TIMEOUT_S = 1800
_PUSH_TIMEOUT_S = 3600
# `git rev-parse --local-env-vars` (git 2.55): a server started from inside a git
# command (a hook, `git exec`) inherits them, and they would point every git
# call here at that other repository instead of the chosen folder.
_GIT_LOCAL_ENV = frozenset(
    {
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_CONFIG",
        "GIT_CONFIG_PARAMETERS",
        "GIT_CONFIG_COUNT",
        "GIT_OBJECT_DIRECTORY",
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_IMPLICIT_WORK_TREE",
        "GIT_GRAFT_FILE",
        "GIT_INDEX_FILE",
        "GIT_NO_REPLACE_OBJECTS",
        "GIT_REPLACE_REF_BASE",
        "GIT_PREFIX",
        "GIT_SHALLOW_FILE",
        "GIT_COMMON_DIR",
    }
)
_ABSOLUTE_REQUIRED = "An absolute folder path is required"
# What `dvc init` writes; `prepare_repository` commits exactly these.
_INIT_FILES = (".dvc/.gitignore", ".dvc/config", ".dvcignore")

State = Literal["ready", "missing", "not_dir", "not_git", "no_dvc"]
Tool = Literal["git", "dvc"]
# One command: arguments, working folder, timeout in seconds -> its result.
Runner = Callable[[list[str], Path, float], subprocess.CompletedProcess[str]]


class InvalidRepositoryPathError(ValueError):
    """The repository is not an absolute folder path."""


class RepositoryNotReadyError(ValueError):
    """The folder is not a git repository with DVC initialized."""


class ToolMissingError(RuntimeError):
    """git or DVC is not installed where the server can run it."""

    def __init__(self, tool: Tool) -> None:
        super().__init__(f"{tool} is not available")
        self.tool = tool


class DvcExportError(RuntimeError):
    """A git or DVC command failed; its output goes to the log, not here."""

    def __init__(self, tool: Tool, step: str) -> None:
        super().__init__(f"{tool} {step} failed")
        self.tool = tool
        self.step = step


def run_command(
    args: list[str], cwd: Path, timeout: float
) -> subprocess.CompletedProcess[str]:
    """Run a tool with no terminal: no prompt can wait for an answer."""
    env = {
        name: value
        for name, value in os.environ.items()
        if name.upper() not in _GIT_LOCAL_ENV
    }
    env.update(GIT_TERMINAL_PROMPT="0", DVC_NO_ANALYTICS="1")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    # The tools are git and DVC and every argument is built in this module.
    return subprocess.run(
        args,
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        creationflags=flags,
        check=False,
    )


@dataclass(frozen=True)
class Tools:
    """The command lines that start git and DVC, ``None`` when missing."""

    git: list[str] | None
    dvc: list[str] | None
    run: Runner = field(default=run_command)


def find_tools() -> Tools:
    """DVC from this environment (the ``[dvc]`` extra) or the PATH; git from the PATH."""
    if importlib.util.find_spec("dvc") is not None:
        dvc: list[str] | None = [sys.executable, "-m", "dvc"]
    else:
        found = shutil.which("dvc")
        dvc = [found] if found else None
    git = shutil.which("git")
    return Tools(git=[git] if git else None, dvc=dvc)


def checked_repository(value: str) -> Path:
    """``value`` as an absolute folder path that is not a drive's root."""
    if not isinstance(value, str) or not value.strip() or len(value) > _MAX_PATH:
        raise InvalidRepositoryPathError(_ABSOLUTE_REQUIRED)
    if "\0" in value:
        raise InvalidRepositoryPathError(_ABSOLUTE_REQUIRED)
    path = Path(value.strip())
    if not path.is_absolute() or path == Path(path.anchor):
        raise InvalidRepositoryPathError(_ABSOLUTE_REQUIRED)
    return path.resolve()


def repository_state(repo: Path) -> State:
    """Whether ``repo`` is a git repository with DVC initialized at its root."""
    if not repo.exists():
        return "missing"
    if not repo.is_dir():
        return "not_dir"
    if not (repo / ".git").exists():
        return "not_git"
    if not (repo / ".dvc").is_dir():
        return "no_dvc"
    return "ready"


def describe_repository(repo: str, tools: Tools | None = None) -> dict[str, Any]:
    """The folder's state, its default DVC remote and which tools exist."""
    tools = tools or find_tools()
    path = checked_repository(repo)
    state = repository_state(path)
    remote = _remote(tools, path) if state == "ready" else None
    return {
        "path": str(path),
        "state": state,
        "remote": remote,
        "tools": {"git": tools.git is not None, "dvc": tools.dvc is not None},
    }


def preview_run(run_dir: Path, repo: str, tools: Tools | None = None) -> dict[str, Any]:
    """The folder, its files (git or DVC) and the README; nothing is written."""
    tools = tools or find_tools()
    repository = describe_repository(repo, tools)
    package = _package(run_dir)
    files = [
        {"path": path, "size": len(data), "storage": "git"}
        for path, data in package.files.items()
    ]
    if package.weights is not None:
        files.append(
            {
                "path": package.weights_name,
                "size": package.weights.stat().st_size,
                "storage": "dvc",
            }
        )
    return {
        "folder": package.folder,
        "files": files,
        "readme": package.files[README].decode("utf-8"),
        "digest": _digest(Path(repository["path"]), package),
        "repository": repository,
    }


def prepare_repository(repo: str, tools: Tools | None = None) -> dict[str, Any]:
    """Make ``repo`` a git repository with DVC, committing what ``dvc init`` wrote.

    A missing folder is created when its parent exists. A folder that is
    already ready is left as it is.
    """
    tools = tools or find_tools()
    path = checked_repository(repo)
    state = repository_state(path)
    if state == "not_dir":
        raise RepositoryNotReadyError("The path is a file, not a folder")
    if state == "ready":
        return describe_repository(repo, tools)
    git, dvc = _require(tools)
    if state == "missing":
        if not path.parent.is_dir():
            raise RepositoryNotReadyError("The parent folder does not exist")
        path.mkdir()
    if not (path / ".git").exists():
        _run(tools, "git", [*git, "init", "-q"], path, "init")
    _run(tools, "dvc", [*dvc, "init", "-q"], path, "init")
    message = "Initialize DVC for VisionForge runs"
    _run(
        tools,
        "git",
        [*git, "commit", "-q", "-m", message, "--", *_INIT_FILES],
        path,
        "commit",
    )
    return describe_repository(repo, tools)


def export_run(
    run_dir: Path, repo: str, digest: str, tools: Tools | None = None
) -> dict[str, Any]:
    """Write the previewed folder, ``dvc add`` the weights, commit, and push.

    Raises:
        InvalidRepositoryPathError: the repository is not an absolute path.
        ToolMissingError: git or DVC is not installed.
        RepositoryNotReadyError: the folder is not a git repository with DVC.
        PreviewMismatchError: the files differ from the confirmed preview.
        ValueError: the run cannot be packaged.
        DvcExportError: a git or DVC command failed.
    """
    tools = tools or find_tools()
    path = checked_repository(repo)
    git, dvc = _require(tools)
    if repository_state(path) != "ready":
        raise RepositoryNotReadyError("The folder is not a git repository with DVC")
    package = _package(run_dir)
    if digest != _digest(path, package):
        raise PreviewMismatchError("Export changed; review and confirm a new preview")
    target = _write(path, package)
    folder = package.folder
    if package.weights is not None:
        weights = f"{folder}/{package.weights_name}"
        shutil.copyfile(package.weights, target / str(package.weights_name))
        _run(tools, "dvc", [*dvc, "add", "-q", weights], path, "add", _ADD_TIMEOUT_S)
    _run(tools, "git", [*git, "add", "--", folder], path, "add")
    commit = _commit(tools, git, path, folder)
    remote = _remote(tools, path)
    pushed: bool | None = None
    if remote is not None and package.weights is not None:
        pointer = f"{folder}/{package.weights_name}.dvc"
        done = _call(tools, [*dvc, "push", "-q", pointer], path, _PUSH_TIMEOUT_S)
        pushed = done is not None and done.returncode == 0
        if not pushed:
            _log("dvc", "push", done)
    return {
        "folder": folder,
        "path": str(target),
        "commit": commit,
        "remote": remote,
        "pushed": pushed,
    }


@dataclass(frozen=True)
class _Package:
    folder: str
    files: dict[str, bytes]
    weights: Path | None
    weights_name: str | None


def _package(run_dir: Path) -> _Package:
    """The run's folder, the files git keeps and the weights DVC keeps."""
    root, data = read_run_record(run_dir)
    weights = _checkpoint(root, data)
    name = f"model{weights.suffix.lower()}" if weights is not None else None
    line = (
        f"- `{name}`: the weights, versioned with DVC (`{name}.dvc`); "
        "`dvc pull` brings them."
        if name
        else "- This run recorded no weights."
    )
    files = report_files(root, data, model=name, weights=line)
    files.update(_dvc_files(data))
    return _Package(run_folder(root, data), files, weights, name)


def _checkpoint(root: Path, data: dict[str, Any]) -> Path | None:
    """The run's selected checkpoint, or ``None`` when it recorded none."""
    artifacts = data.get("artifacts")
    model = artifacts.get("model") if isinstance(artifacts, dict) else None
    if not isinstance(model, str) or not model:
        return None
    path = contained_run_file(root, model)
    if path.suffix.lower() not in _WEIGHTS:
        raise ValueError("Unsupported checkpoint format")
    return path


def _dvc_files(data: dict[str, Any]) -> dict[str, bytes]:
    """params.yaml, metrics.json, history.csv and the dvc.yaml naming them."""
    files: dict[str, bytes] = {}
    declared: dict[str, Any] = {}
    config = public_record(data).get("config")
    if isinstance(config, dict) and config:
        text = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
        files[PARAMS_FILE] = text.encode("utf-8")
        declared["params"] = [PARAMS_FILE]
    metrics = numeric_values(data.get("metrics"))
    if metrics:
        text = json.dumps(metrics, indent=2, sort_keys=True) + "\n"
        files[METRICS_FILE] = text.encode("utf-8")
        declared["metrics"] = [METRICS_FILE]
    history = _history_csv(data.get("history"))
    if history is not None:
        files[HISTORY_FILE] = history
        declared["plots"] = [{HISTORY_FILE: {"x": "epoch"}}]
    if declared:
        files[DVC_FILE] = yaml.safe_dump(declared, sort_keys=False).encode("utf-8")
    return files


def _history_csv(history: Any) -> bytes | None:
    """One row per epoch with every numeric column; ``None`` without numbers."""
    rows = [numeric_values(row) for row in history] if isinstance(history, list) else []
    columns = list(dict.fromkeys(key for row in rows for key in row if key != "epoch"))
    if not columns:
        return None
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(["epoch", *columns])
    for index, row in enumerate(rows, start=1):
        epoch = row.get("epoch", index)
        writer.writerow([_plain(epoch), *(_plain(row.get(key)) for key in columns)])
    return out.getvalue().encode("utf-8")


def _plain(value: float | None) -> str:
    if value is None:
        return ""
    return str(int(value)) if float(value).is_integer() else repr(float(value))


def _digest(repo: Path, package: _Package) -> str:
    listing = {
        path: hashlib.sha256(data).hexdigest() for path, data in package.files.items()
    }
    if package.weights is not None:
        listing[str(package.weights_name)] = _file_sha256(package.weights)
    text = json.dumps(
        {"repository": str(repo), "folder": package.folder, "files": listing},
        sort_keys=True,
    )
    return hashlib.sha256(text.encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _write(repo: Path, package: _Package) -> Path:
    """Write the git files under ``repo/<folder>``; the folder stays inside."""
    target = repo / package.folder
    # Checked before anything is created: a linked `runs` could point elsewhere.
    if not target.resolve().is_relative_to(repo):
        raise ValueError("The run folder would leave the repository")
    target.mkdir(parents=True, exist_ok=True)
    for name, data in package.files.items():
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    return target


def _commit(tools: Tools, git: list[str], repo: Path, folder: str) -> str | None:
    """Commit only ``folder``; ``None`` when nothing in it changed."""
    staged = _call(tools, [*git, "diff", "--cached", "--quiet", "--", folder], repo)
    if staged is not None and staged.returncode == 0:
        return None
    if staged is None or staged.returncode != 1:
        _log("git", "diff", staged)
        raise DvcExportError("git", "diff")
    message = f"Add VisionForge run {folder}"
    _run(
        tools,
        "git",
        [*git, "commit", "-q", "-m", message, "--", folder],
        repo,
        "commit",
    )
    head = _run(tools, "git", [*git, "rev-parse", "HEAD"], repo, "rev-parse")
    return head.stdout.strip() or None


def _remote(tools: Tools, repo: Path) -> str | None:
    """The repository's default DVC remote; ``None`` without one or without DVC."""
    if tools.dvc is None:
        return None
    done = _call(tools, [*tools.dvc, "remote", "default"], repo)
    if done is None or done.returncode != 0:
        return None
    name = done.stdout.strip()
    return name or None


def _require(tools: Tools) -> tuple[list[str], list[str]]:
    if tools.git is None:
        raise ToolMissingError("git")
    if tools.dvc is None:
        raise ToolMissingError("dvc")
    return tools.git, tools.dvc


def _run(
    tools: Tools,
    tool: Tool,
    args: list[str],
    cwd: Path,
    step: str,
    timeout: float = _GIT_TIMEOUT_S,
) -> subprocess.CompletedProcess[str]:
    done = _call(tools, args, cwd, timeout)
    if done is None or done.returncode != 0:
        _log(tool, step, done)
        raise DvcExportError(tool, step)
    return done


def _call(
    tools: Tools, args: list[str], cwd: Path, timeout: float = _GIT_TIMEOUT_S
) -> subprocess.CompletedProcess[str] | None:
    """The command's result, or ``None`` when it could not start or timed out."""
    try:
        return tools.run(args, cwd, timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("DVC export: {} did not finish: {}", Path(args[0]).name, exc)
        return None


def _log(tool: Tool, step: str, done: subprocess.CompletedProcess[str] | None) -> None:
    output = (done.stderr or done.stdout).strip()[-2000:] if done is not None else ""
    code = done.returncode if done is not None else None
    logger.warning("DVC export: {} {} exited {}: {}", tool, step, code, output)
