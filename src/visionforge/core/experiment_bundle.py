"""Export bounded experiment records without copying arbitrary output trees."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import IO, Any

_MAX_MEMBERS = 4096
_MAX_MANIFEST_BYTES = 4 * 1024 * 1024
_MAX_RUN_BYTES = 16 * 1024 * 1024
_MAX_TOTAL_BYTES = 16 * 1024**3
_CHUNK_SIZE = 1024 * 1024
_EXTENSIONS = {
    "model": {".pt", ".pth", ".onnx", ".safetensors"},
    "weights": {".pt", ".pth", ".onnx", ".safetensors"},
    "checkpoint": {".pt", ".pth", ".safetensors"},
    "config": {".json", ".yaml", ".yml"},
    "history": {".json", ".csv", ".jsonl"},
    "predictions": {".json", ".csv", ".jsonl", ".parquet", ".npz"},
}
_CONVENTIONAL = (
    "config.json",
    "config.yaml",
    "config.yml",
    "history.json",
    "history.csv",
    "predictions.json",
    "predictions.csv",
    "best_model.pth",
    "weights/best.pt",
    "weights/last.pt",
)
_PRIVATE_NAMES = {
    "data",
    "dataset",
    "datasets",
    "user_models",
    "user_tasks",
    "secrets",
    "secret",
    "credentials",
    "credential",
    "tokens",
    "token",
    "keys",
    "key",
}


def _safe_name(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(
        name
        and "\\" not in name
        and ":" not in name
        and "\x00" not in name
        and not path.is_absolute()
        and not PureWindowsPath(name).is_absolute()
        and all(part not in {"", ".", ".."} for part in name.split("/"))
    )


def _contained_file(root: Path, value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        local = root / candidate
        # Older run.json files store paths relative to the process workspace.
        candidate = (
            candidate.absolute() if not local.exists() and candidate.exists() else local
        )
    resolved = candidate.resolve()
    relative = resolved.relative_to(root)
    if not _safe_name(relative.as_posix()):
        raise ValueError("unsafe artifact path")
    if any(
        part.startswith(".") or Path(part).stem.lower() in _PRIVATE_NAMES
        for part in relative.parts
    ):
        raise ValueError("private or dataset artifact")
    current = candidate
    while current != root and current != current.parent:
        if current.is_symlink() or current.is_junction():
            raise ValueError("linked artifact")
        current = current.parent
    if not resolved.is_file():
        raise FileNotFoundError("artifact unavailable")
    return resolved


def _copy_and_hash(source: IO[bytes], target: IO[bytes]) -> dict[str, int | str]:
    digest = hashlib.sha256()
    size = 0
    while chunk := source.read(_CHUNK_SIZE):
        size += len(chunk)
        if size > _MAX_TOTAL_BYTES:
            raise ValueError("bundle exceeds size limit")
        digest.update(chunk)
        target.write(chunk)
    return {"size": size, "sha256": digest.hexdigest()}


def export_experiment_bundle(run_dir: Path | str, destination: Path | str) -> Path:
    """Write a ZIP experiment record and refuse an existing destination.

    Raises:
        FileExistsError: The destination already exists.
        FileNotFoundError: The selected run has no local run.json.
        ValueError: The run record is invalid or exceeds the bundle limits.
        OSError: Writing or atomic publication is unsupported or fails.
    """
    root = Path(run_dir).resolve(strict=True)
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    run_path = _contained_file(root, "run.json")
    with run_path.open("rb") as handle:
        raw = handle.read(_MAX_RUN_BYTES + 1)
    if len(raw) > _MAX_RUN_BYTES:
        raise ValueError("run.json exceeds size limit")
    data = json.loads(raw)
    if not isinstance(data, dict) or not isinstance(data.get("artifacts", {}), dict):
        raise ValueError("run.json must contain an object with an artifacts object")
    files = {"run.json": run_path}
    notes: list[dict[str, str]] = []
    for category, values in data.get("artifacts", {}).items():
        label = (
            category
            if category in {*_EXTENSIONS, "graphics", "report"}
            else "unsupported"
        )
        if values is None:
            notes.append({"artifact": label, "reason": "not recorded"})
            continue
        values = values if isinstance(values, list) else [values]
        for value in values:
            if category not in _EXTENSIONS or not isinstance(value, str):
                notes.append(
                    {
                        "artifact": label,
                        "reason": "unsupported artifact category or value",
                    }
                )
                continue
            if Path(value).suffix.lower() not in _EXTENSIONS[category]:
                notes.append({"artifact": category, "reason": "unsupported file type"})
                continue
            try:
                path = _contained_file(root, value)
            except FileNotFoundError:
                notes.append({"artifact": category, "reason": "missing local file"})
            except (ValueError, OSError):
                notes.append(
                    {"artifact": category, "reason": "outside run or linked file"}
                )
            else:
                name = path.relative_to(root).as_posix()
                if name.casefold() == "manifest.json":
                    notes.append(
                        {"artifact": category, "reason": "reserved bundle member name"}
                    )
                    continue
                files[name] = path
    for name in _CONVENTIONAL:
        try:
            files[name] = _contained_file(root, name)
        except FileNotFoundError:
            continue
        except (ValueError, OSError):
            notes.append({"artifact": name, "reason": "outside run or linked file"})
    if len(files) + 1 > _MAX_MEMBERS:
        raise ValueError("bundle exceeds member limit")
    descriptor, temporary = tempfile.mkstemp(
        prefix=".visionforge-bundle-", suffix=".zip", dir=destination.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        records: dict[str, dict[str, int | str]] = {
            "run.json": {"size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        }
        total_size = len(raw)
        with zipfile.ZipFile(
            temporary_path, "w", compression=zipfile.ZIP_DEFLATED
        ) as bundle:
            bundle.writestr("run.json", raw)
            for name, source in sorted(files.items()):
                if name == "run.json":
                    continue
                with (
                    source.open("rb") as handle,
                    bundle.open(name, "w", force_zip64=True) as target,
                ):
                    if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                        raise ValueError("artifact is not a regular file")
                    records[name] = _copy_and_hash(handle, target)
                total_size += int(records[name]["size"])
                if total_size > _MAX_TOTAL_BYTES:
                    raise ValueError("bundle exceeds size limit")
            manifest = json.dumps(
                {"format_version": 1, "files": records, "notes": notes}, indent=2
            ).encode()
            if len(manifest) > _MAX_MANIFEST_BYTES:
                raise ValueError("bundle manifest exceeds size limit")
            if total_size + len(manifest) > _MAX_TOTAL_BYTES:
                raise ValueError("bundle exceeds size limit")
            bundle.writestr("manifest.json", manifest)
        # A hard link publishes the completed file atomically without replacing a competitor.
        os.link(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination


def verify_experiment_bundle(bundle_path: Path | str) -> dict[str, Any]:
    """Validate archive members and SHA256 checksums without extracting files.

    Raises:
        ValueError: The ZIP, manifest, paths, sizes or checksums are invalid.
    """
    try:
        with zipfile.ZipFile(bundle_path) as bundle:
            members = bundle.infolist()
            names = [member.filename for member in members]
            if len(members) > _MAX_MEMBERS or len(set(names)) != len(names):
                raise ValueError("bundle member count exceeds limit or duplicates")
            if any(not _safe_name(member.orig_filename) for member in members):
                raise ValueError("unsafe bundle member path")
            if any(
                stat.S_ISLNK(member.external_attr >> 16) or member.is_dir()
                for member in members
            ):
                raise ValueError("linked or directory bundle member")
            if sum(member.file_size for member in members) > _MAX_TOTAL_BYTES:
                raise ValueError("bundle exceeds size limit")
            if "manifest.json" not in names or "run.json" not in names:
                raise ValueError("missing bundle metadata")
            if bundle.getinfo("manifest.json").file_size > _MAX_MANIFEST_BYTES:
                raise ValueError("bundle manifest exceeds size limit")
            manifest = json.loads(bundle.read("manifest.json"))
            if not isinstance(manifest, dict) or manifest.get("format_version") != 1:
                raise ValueError("unsupported bundle manifest")
            records = manifest.get("files")
            if not isinstance(records, dict) or set(records) != set(names) - {
                "manifest.json"
            }:
                raise ValueError("manifest does not match bundle members")
            for name, record in records.items():
                if (
                    not isinstance(record, dict)
                    or record.get("size") != bundle.getinfo(name).file_size
                ):
                    raise ValueError("bundle integrity check failed: size")
                digest = hashlib.sha256()
                with bundle.open(name) as handle:
                    while chunk := handle.read(_CHUNK_SIZE):
                        digest.update(chunk)
                if record.get("sha256") != digest.hexdigest():
                    raise ValueError("bundle integrity check failed: SHA256")
            return manifest
    except (
        zipfile.BadZipFile,
        KeyError,
        TypeError,
        UnicodeError,
        json.JSONDecodeError,
        RuntimeError,
        NotImplementedError,
    ) as exc:
        raise ValueError("invalid experiment bundle") from exc


__all__ = ["export_experiment_bundle", "verify_experiment_bundle"]
