"""Export bounded experiment records without copying arbitrary output trees."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import IO, Any

SAFETENSORS_SUFFIX = ".safetensors"
JSON_SUFFIX = ".json"
BUNDLE_SIZE_LIMIT_ERROR = "bundle exceeds size limit"
RUN_FILENAME = "run.json"
MANIFEST_FILENAME = "manifest.json"

_MAX_MEMBERS = 4096
_MAX_MANIFEST_BYTES = 4 * 1024 * 1024
_MAX_RUN_BYTES = 16 * 1024 * 1024
_MAX_TOTAL_BYTES = 16 * 1024**3
# An imported run is a checkpoint and its records, not a dataset: well under
# the export ceiling, both for the archive and for what it unpacks to.
_MAX_IMPORT_BYTES = 2 * 1024**3
_CHUNK_SIZE = 1024 * 1024
_EXTENSIONS = {
    "model": {".pt", ".pth", ".onnx", SAFETENSORS_SUFFIX},
    "weights": {".pt", ".pth", ".onnx", SAFETENSORS_SUFFIX},
    "checkpoint": {".pt", ".pth", SAFETENSORS_SUFFIX},
    "config": {JSON_SUFFIX, ".yaml", ".yml"},
    "history": {JSON_SUFFIX, ".csv", ".jsonl"},
    "predictions": {JSON_SUFFIX, ".csv", ".jsonl", ".parquet", ".npz"},
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
            raise ValueError(BUNDLE_SIZE_LIMIT_ERROR)
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
    run_path, raw, data = _read_run_record(root)
    files = {RUN_FILENAME: run_path}
    notes: list[dict[str, str]] = []
    _collect_recorded_artifacts(root, data, files, notes)
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
        _write_bundle_archive(temporary_path, raw, files, notes)
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
            names = _validate_bundle_members(bundle)
            manifest = json.loads(bundle.read(MANIFEST_FILENAME))
            if not isinstance(manifest, dict) or manifest.get("format_version") != 1:
                raise ValueError("unsupported bundle manifest")
            records = manifest.get("files")
            if not isinstance(records, dict) or set(records) != set(names) - {
                MANIFEST_FILENAME
            }:
                raise ValueError("manifest does not match bundle members")
            _verify_bundle_records(bundle, records)
            return manifest
    except (
        zipfile.BadZipFile,
        KeyError,
        TypeError,
        UnicodeError,
        json.JSONDecodeError,
        RuntimeError,
    ) as exc:
        raise ValueError("invalid experiment bundle") from exc


def _read_run_record(root: Path) -> tuple[Path, bytes, dict[str, Any]]:
    """Read and validate the bounded run record before collecting artifacts."""
    run_path = _contained_file(root, RUN_FILENAME)
    with run_path.open("rb") as handle:
        raw = handle.read(_MAX_RUN_BYTES + 1)
    if len(raw) > _MAX_RUN_BYTES:
        raise ValueError("run.json exceeds size limit")
    data = json.loads(raw)
    if not isinstance(data, dict) or not isinstance(data.get("artifacts", {}), dict):
        raise ValueError("run.json must contain an object with an artifacts object")
    return run_path, raw, data


def _collect_recorded_artifacts(
    root: Path,
    data: dict[str, Any],
    files: dict[str, Path],
    notes: list[dict[str, str]],
) -> None:
    """Collect supported recorded artifacts while retaining omission notes."""
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
            _collect_artifact(root, category, label, value, files, notes)


def _collect_artifact(
    root: Path,
    category: str,
    label: str,
    value: Any,
    files: dict[str, Path],
    notes: list[dict[str, str]],
) -> None:
    """Validate one artifact and preserve its original omission reason."""
    if category not in _EXTENSIONS or not isinstance(value, str):
        notes.append(
            {
                "artifact": label,
                "reason": "unsupported artifact category or value",
            }
        )
        return
    if Path(value).suffix.lower() not in _EXTENSIONS[category]:
        notes.append({"artifact": category, "reason": "unsupported file type"})
        return
    try:
        path = _contained_file(root, value)
    except FileNotFoundError:
        notes.append({"artifact": category, "reason": "missing local file"})
    except (ValueError, OSError):
        notes.append({"artifact": category, "reason": "outside run or linked file"})
    else:
        name = path.relative_to(root).as_posix()
        if name.casefold() == MANIFEST_FILENAME:
            notes.append(
                {"artifact": category, "reason": "reserved bundle member name"}
            )
            return
        files[name] = path


def _write_bundle_archive(
    temporary_path: Path,
    raw: bytes,
    files: dict[str, Path],
    notes: list[dict[str, str]],
) -> None:
    """Write bounded artifact contents and their integrity manifest to a ZIP."""
    records: dict[str, dict[str, int | str]] = {
        RUN_FILENAME: {"size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    }
    total_size = len(raw)
    with zipfile.ZipFile(
        temporary_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as bundle:
        bundle.writestr(RUN_FILENAME, raw)
        for name, source in sorted(files.items()):
            if name == RUN_FILENAME:
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
                raise ValueError(BUNDLE_SIZE_LIMIT_ERROR)
        manifest = json.dumps(
            {"format_version": 1, "files": records, "notes": notes}, indent=2
        ).encode()
        if len(manifest) > _MAX_MANIFEST_BYTES:
            raise ValueError("bundle manifest exceeds size limit")
        if total_size + len(manifest) > _MAX_TOTAL_BYTES:
            raise ValueError(BUNDLE_SIZE_LIMIT_ERROR)
        bundle.writestr(MANIFEST_FILENAME, manifest)


def _validate_bundle_members(bundle: zipfile.ZipFile) -> list[str]:
    """Reject unsafe archive members before reading bundle metadata."""
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
        raise ValueError(BUNDLE_SIZE_LIMIT_ERROR)
    if MANIFEST_FILENAME not in names or RUN_FILENAME not in names:
        raise ValueError("missing bundle metadata")
    if bundle.getinfo(MANIFEST_FILENAME).file_size > _MAX_MANIFEST_BYTES:
        raise ValueError("bundle manifest exceeds size limit")
    return names


def _verify_bundle_records(bundle: zipfile.ZipFile, records: dict[str, Any]) -> None:
    """Verify recorded member sizes and streaming SHA256 checksums."""
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


def import_experiment_bundle(
    bundle_path: Path | str,
    models_dir: Path | str,
    origin: dict[str, Any] | None = None,
) -> Path:
    """Unpack a verified bundle as a run of ``models_dir`` and return its folder.

    The same import serves a bundle a Colab job brought back (ADR-163, phase 3)
    and one the researcher exported elsewhere and downloaded by hand. The run is
    assembled in a staging folder next to ``models_dir``, out of the History
    scan, and moved into ``models_dir/<experiment>/<run folder>`` in one rename:
    a failure leaves nothing half-written among the runs. Paths that run.json
    recorded on the other machine are rewritten to the files that arrived;
    anything the bundle did not carry is dropped rather than left pointing at a
    folder that does not exist here.

    Raises:
        FileExistsError: A run with that experiment and folder is already here.
        ValueError: The bundle is invalid, too large or not a run record.
        OSError: Reading, writing or the final rename fails.
    """
    bundle_path = Path(bundle_path)
    if bundle_path.stat().st_size > _MAX_IMPORT_BYTES:
        raise ValueError(BUNDLE_SIZE_LIMIT_ERROR)
    manifest = verify_experiment_bundle(bundle_path)
    root = Path(models_dir).resolve()
    with zipfile.ZipFile(bundle_path) as bundle:
        if sum(info.file_size for info in bundle.infolist()) > _MAX_IMPORT_BYTES:
            raise ValueError(BUNDLE_SIZE_LIMIT_ERROR)
        data = _imported_run_record(bundle.read(RUN_FILENAME))
        experiment: str = data["experiment"]
        target: Path = root / experiment / _run_folder_name(data)
        if target.exists():
            raise FileExistsError(target)
        root.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".visionforge-import-", dir=root.parent))
        try:
            arrived = _extract_members(bundle, set(manifest["files"]), staging)
            _rebase_run_record(data, target, arrived)
            if origin is not None:
                data["origin"] = origin
            (staging / RUN_FILENAME).write_text(
                json.dumps(data, indent=2), encoding="utf-8"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise FileExistsError(target)
            os.rename(staging, target)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    return target


def _plain_name(value: Any) -> bool:
    """One path component a run folder may use: no separator, no dot file."""
    return (
        isinstance(value, str)
        and _safe_name(value)
        and "/" not in value
        and not value.startswith(".")
        and len(value) <= 200
    )


def _imported_run_record(raw: bytes) -> dict[str, Any]:
    """The bundle's run.json, checked for what History needs to list it."""
    if len(raw) > _MAX_RUN_BYTES:
        raise ValueError("run.json exceeds size limit")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("run.json must contain an object")
    config = data.get("config")
    metrics = data.get("metrics")
    if not (
        _plain_name(data.get("experiment"))
        and isinstance(data.get("status"), str)
        and isinstance(data.get("timestamp"), str)
        and isinstance(config, dict)
        and isinstance(metrics, dict)
        and isinstance(metrics.get("total_epochs"), int)
    ):
        raise ValueError("run.json is not a complete run record")
    version = config.get("schema_version", 1)
    if not isinstance(version, int) or version < 1:
        raise ValueError("run.json has an invalid schema_version")
    # Lazy import: core.experiment_bundle stays usable without the config models.
    from visionforge.utils.config import check_schema_version

    check_schema_version(version)
    return data


def _run_folder_name(data: dict[str, Any]) -> str:
    """The run's own folder name when it recorded a usable one, else a new stamp."""
    recorded = data.get("run_dir")
    if isinstance(recorded, str):
        name = PurePosixPath(recorded.replace("\\", "/")).name
        if _plain_name(name):
            return name
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _extract_members(
    bundle: zipfile.ZipFile, names: set[str], staging: Path
) -> set[str]:
    """Copy every member but run.json and the manifest; names were verified safe."""
    arrived: set[str] = set()
    total = 0
    for name in sorted(names - {RUN_FILENAME}):
        destination = staging.joinpath(*name.split("/"))
        if not destination.resolve().is_relative_to(staging.resolve()):
            raise ValueError("unsafe bundle member path")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with bundle.open(name) as source, destination.open("xb") as target:
            while chunk := source.read(_CHUNK_SIZE):
                total += len(chunk)
                if total > _MAX_IMPORT_BYTES:
                    raise ValueError(BUNDLE_SIZE_LIMIT_ERROR)
                target.write(chunk)
        arrived.add(name)
    return arrived


def _rebase_run_record(data: dict[str, Any], target: Path, arrived: set[str]) -> None:
    """Point run.json at the files as they now are on this machine."""
    # The run may come from a Linux VM or a Windows machine: compare as text with
    # one separator, so a drive path is understood on any OS.
    origin_dir = str(data.get("run_dir", "")).replace("\\", "/").rstrip("/")
    data["run_dir"] = str(target)

    def local(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        text = value.replace("\\", "/")
        if origin_dir and text.startswith(origin_dir + "/"):
            text = text[len(origin_dir) + 1 :]
        elif text.startswith("/") or PureWindowsPath(value).is_absolute():
            return None
        if not _safe_name(text) or text not in arrived:
            return None
        return str(target.joinpath(*text.split("/")))

    artifacts = data.get("artifacts")
    if not isinstance(artifacts, dict):
        data["artifacts"] = {"model": None, "graphics": [], "report": None}
        return
    for category, value in artifacts.items():
        if isinstance(value, list):
            artifacts[category] = [p for p in map(local, value) if p is not None]
        elif value is not None:
            artifacts[category] = local(value)


__all__ = [
    "export_experiment_bundle",
    "import_experiment_bundle",
    "verify_experiment_bundle",
]
