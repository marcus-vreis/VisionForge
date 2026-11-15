"""Preview and explicitly export a bounded model package to a private HF repository."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from visionforge.core.tracking_privacy import public_metrics

_ENDPOINT = "https://huggingface.co"
_MAX_RECORD = 16 * 1024 * 1024
_EXTENSIONS = {".pt", ".pth", ".onnx", ".safetensors"}
_RECORD = "run.json"
_MANIFEST = "manifest.json"
# Two patterns, one condition: a private word anywhere, or a short one as a
# whole ``_``/``-`` separated part (``logs`` but not ``catalog``).
_PRIVATE_WORDS = re.compile(r"config|secret|credential|token|password|dataset", re.I)
_PRIVATE_PARTS = re.compile(r"(?:^|[_-])(?:logs?|data|keys?)(?:$|[_-])", re.I)
_TASKS = {"classification", "detection", "regression", "segmentation", "anomaly"}


def _is_private(stem: str) -> bool:
    return bool(_PRIVATE_WORDS.search(stem) or _PRIVATE_PARTS.search(stem))


class HubExportError(RuntimeError):
    """Report an optional dependency or remote export failure without leaking secrets."""


class PreviewMismatchError(ValueError):
    """Reject a package that differs from the user's confirmed preview."""


class PrivateRepositoryRequiredError(ValueError):
    """Reject a repository whose private visibility cannot be confirmed."""


def preview_run(run_dir: Path) -> dict[str, Any]:
    """Describe the exact safe package without credentials or network access."""
    with tempfile.TemporaryDirectory(prefix="visionforge-hub-") as temporary:
        return _prepare(run_dir, Path(temporary))


def export_run(run_dir: Path, repo_id: str, token: str, digest: str) -> dict[str, Any]:
    """Upload the confirmed package after verifying the official destination is private."""
    if not isinstance(token, str) or not token.strip():
        raise ValueError("An explicit Hugging Face token is required")
    if not is_valid_repo_id(repo_id):
        raise ValueError("Repository must be an explicit owner/name")
    with tempfile.TemporaryDirectory(prefix="visionforge-hub-") as temporary:
        staging = Path(temporary)
        preview = _prepare(run_dir, staging)
        if not isinstance(digest, str) or digest != preview["digest"]:
            raise PreviewMismatchError(
                "Export changed; review and confirm a new preview"
            )
        try:
            from huggingface_hub import HfApi

            # Never inherit HF_ENDPOINT or implicit token lookup from the server.
            client = HfApi(endpoint=_ENDPOINT, token=token)
            client.create_repo(
                repo_id, token=token, private=True, repo_type="model", exist_ok=True
            )
            info = client.repo_info(repo_id, token=token, repo_type="model")
            if getattr(info, "private", None) is not True:
                raise PrivateRepositoryRequiredError(
                    "A verified private repository is required"
                )
            client.upload_folder(
                repo_id=repo_id,
                folder_path=staging,
                token=token,
                repo_type="model",
                commit_message="Export VisionForge model",
                allow_patterns=preview["files"],
            )
        except PrivateRepositoryRequiredError:
            raise
        except ImportError:
            raise HubExportError(
                "Install the optional Hugging Face Hub dependency"
            ) from None
        except Exception:  # noqa: BLE001
            # SDK errors can contain tokens or private paths.
            raise HubExportError("Hugging Face Hub export failed") from None
    return {
        "repo_id": repo_id,
        "url": f"{_ENDPOINT}/{repo_id}",
        "files": preview["files"],
    }


def is_valid_repo_id(value: str) -> bool:
    """True for an explicit ``owner/name`` the Hub accepts."""
    if not isinstance(value, str) or len(value) > 193 or value.count("/") != 1:
        return False
    return all(
        re.fullmatch(r"\w[\w.-]{0,94}\w", part, re.ASCII) is not None
        and "--" not in part
        and ".." not in part
        and not part.endswith(".git")
        or re.fullmatch(r"\w", part, re.ASCII) is not None
        for part in value.split("/")
    )


def _safe_file(root: Path, value: str) -> Path:
    candidate = Path(value)
    if ".." in candidate.parts:
        raise ValueError("Unsafe artifact path")
    if not candidate.is_absolute():
        local = root / candidate
        # Native trainers can record outputs/models/... relative to the workspace.
        # The fallback must still pass the same root and link checks below.
        candidate = (
            candidate.absolute()
            if not local.exists()
            and not local.is_symlink()
            and not local.is_junction()
            and candidate.exists()
            else local
        )
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        raise ValueError("Artifact is outside the selected run") from None
    if not relative.parts or any(
        part in {".", ".."} or part.startswith(".") or _is_private(Path(part).stem)
        for part in relative.parts
    ):
        raise ValueError("Unsafe artifact path")
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink() or current.is_junction():
            raise ValueError("Linked artifacts cannot be exported")
    if not candidate.resolve().is_relative_to(root):
        raise ValueError("Artifact is outside the selected run")
    if not candidate.is_file():
        raise FileNotFoundError("Selected artifact is unavailable")
    return candidate


def _prepare(run_dir: Path, staging: Path) -> dict[str, Any]:
    root = run_dir.resolve(strict=True)
    source = _safe_file(root, _RECORD)
    with source.open("rb") as stream:
        raw = stream.read(_MAX_RECORD + 1)
    if len(raw) > _MAX_RECORD:
        raise ValueError("Run record exceeds size limit")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("Invalid run record")
    artifacts = data.get("artifacts")
    model = artifacts.get("model") if isinstance(artifacts, dict) else None
    if not isinstance(model, str) or not model:
        raise ValueError("Run has no selected checkpoint")
    checkpoint = _safe_file(root, model)
    if checkpoint.suffix.lower() not in _EXTENSIONS:
        raise ValueError("Unsupported checkpoint format")
    checkpoint_name = f"weights/model{checkpoint.suffix.lower()}"
    target = staging / checkpoint_name
    target.parent.mkdir()
    _copy_checkpoint(root, model, checkpoint, target)
    config = data.get("config")
    task = _public_task(data, config)
    status = data.get("status")
    status = (
        status
        if isinstance(status, str)
        and status in {"completed", "stopped", "failed", "running"}
        else "unknown"
    )
    history = data.get("history")
    metrics = public_metrics(data.get("metrics"))
    record = {
        "export_format": "visionforge-hub-1",
        "task": task,
        "status": status,
        "metrics": metrics,
        "history": [public_metrics(row) for row in history]
        if isinstance(history, list)
        else [],
    }
    _write_json(staging / _RECORD, record)
    card = (
        "---\ntags:\n- visionforge\n---\n# VisionForge model\n\n"
        f"Task: {task}. Run status: {status}.\n\n"
        f"{_architecture_line(config)}"
        f"Checkpoint: `{checkpoint_name}`.\n\n"
        f"{_metrics_table(metrics)}"
        "This checkpoint requires its original VisionForge task architecture and "
        "preprocessing. It is not a Transformers from_pretrained package.\n\n"
        "run.json is a sanitised metrics projection; datasets, configuration and "
        "local paths are excluded. No license is inferred.\n"
    )
    (staging / "README.md").write_text(card, encoding="utf-8")
    names = ["README.md", _RECORD, checkpoint_name]
    manifest = {
        name: {
            "size": (staging / name).stat().st_size,
            "sha256": _file_digest(staging / name),
        }
        for name in sorted(names)
    }
    _write_json(staging / _MANIFEST, manifest)
    digest = _file_digest(staging / _MANIFEST)
    return {"files": sorted([*names, _MANIFEST]), "digest": digest, "card": card}


def _copy_checkpoint(root: Path, model: str, checkpoint: Path, target: Path) -> None:
    """Copy the checkpoint, refusing it if it was swapped or changed meanwhile."""
    # Check the opened object and all path components before and after copying.
    before = checkpoint.stat()
    with checkpoint.open("rb") as reader, target.open("xb") as writer:
        opened = os.fstat(reader.fileno())
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError("Checkpoint changed while preparing export")
        _safe_file(root, model)
        shutil.copyfileobj(reader, writer)
    after = _safe_file(root, model).stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError("Checkpoint changed while preparing export")


def _public_task(data: dict[str, Any], config: Any) -> str:
    """The run's task family, or ``unknown``; classification modes are merged."""
    task = config.get("task") if isinstance(config, dict) else data.get("task")
    if isinstance(task, str) and task in {"binary", "multiclass", "multilabel"}:
        task = "classification"
    return task if isinstance(task, str) and task in _TASKS else "unknown"


def _architecture_line(config: Any) -> str:
    """``Architecture: `name`.`` for a plain model name; empty otherwise."""
    model = config.get("model") if isinstance(config, dict) else None
    name = model.get("name") if isinstance(model, dict) else None
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", name):
        return ""
    return f"Architecture: `{name}`.\n\n"


def _metrics_table(metrics: dict[str, int | float]) -> str:
    """A Markdown table of the already-filtered final metrics; empty if none."""
    if not metrics:
        return ""
    rows = "".join(f"| {key} | {value:.4g} |\n" for key, value in metrics.items())
    return f"| Metric | Value |\n|---|---|\n{rows}\n"


def _file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, sort_keys=True, indent=2, allow_nan=False), encoding="utf-8"
    )
