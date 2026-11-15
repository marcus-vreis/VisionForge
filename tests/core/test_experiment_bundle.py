"""Experiment exports include only declared, contained run artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import zipfile
from pathlib import Path

import pytest

from visionforge.core.experiment_bundle import (
    export_experiment_bundle,
    verify_experiment_bundle,
)


def _run(tmp_path: Path, artifacts: dict[str, object] | None = None) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(
        json.dumps(
            {"config": {"seed": 42}, "history": [], "artifacts": artifacts or {}}
        ),
        encoding="utf-8",
    )
    return run_dir


def test_exports_declared_artifacts_with_relative_names_and_hashes(
    tmp_path: Path,
) -> None:
    run_dir = _run(
        tmp_path, {"model": "weights/best.pt", "predictions": "predictions.csv"}
    )
    (run_dir / "weights").mkdir()
    (run_dir / "weights/best.pt").write_bytes(b"weights")
    (run_dir / "predictions.csv").write_text("label\ncat\n", encoding="utf-8")
    (run_dir / "secret.env").write_text("private", encoding="utf-8")
    (run_dir / "dataset").mkdir()
    (run_dir / "dataset/image.jpg").write_bytes(b"dataset")
    destination = tmp_path / "bundle.zip"

    assert export_experiment_bundle(run_dir, destination) == destination
    manifest = verify_experiment_bundle(destination)
    with zipfile.ZipFile(destination) as bundle:
        assert set(bundle.namelist()) == {
            "run.json",
            "weights/best.pt",
            "predictions.csv",
            "manifest.json",
        }
        for name, record in manifest["files"].items():
            data = bundle.read(name)
            assert record == {
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }


def test_excludes_external_and_unsupported_artifacts_without_disclosing_paths(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "private.pt"
    outside.write_bytes(b"private")
    run_dir = _run(
        tmp_path,
        {
            "model": str(outside),
            "weights": "../private.pt",
            "config": ".env",
            "predictions": "missing.csv",
            "graphics": ["plot.png"],
        },
    )
    (run_dir / ".env").write_text("TOKEN=private", encoding="utf-8")
    destination = tmp_path / "bundle.zip"
    export_experiment_bundle(run_dir, destination)
    manifest = verify_experiment_bundle(destination)
    assert len(manifest["notes"]) >= 5
    assert str(tmp_path) not in json.dumps(manifest)
    with zipfile.ZipFile(destination) as bundle:
        assert set(bundle.namelist()) == {"run.json", "manifest.json"}


def test_accepts_absolute_artifact_only_inside_run(tmp_path: Path) -> None:
    run_dir = _run(tmp_path)
    model = run_dir / "best_model.pth"
    model.write_bytes(b"model")
    (run_dir / "run.json").write_text(json.dumps({"artifacts": {"model": str(model)}}))
    destination = tmp_path / "bundle.zip"
    export_experiment_bundle(run_dir, destination)
    assert "best_model.pth" in verify_experiment_bundle(destination)["files"]


def test_refuses_existing_destination_and_preserves_it(tmp_path: Path) -> None:
    run_dir = _run(tmp_path)
    destination = tmp_path / "bundle.zip"
    destination.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        export_experiment_bundle(run_dir, destination)
    assert destination.read_bytes() == b"original"


@pytest.mark.parametrize("content", [None, "{", "[]"])
def test_missing_or_invalid_run_json_leaves_no_destination(
    tmp_path: Path, content: str | None
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    if content is not None:
        (run_dir / "run.json").write_text(content, encoding="utf-8")
    destination = tmp_path / "bundle.zip"
    with pytest.raises((FileNotFoundError, ValueError)):
        export_experiment_bundle(run_dir, destination)
    assert not destination.exists()


@pytest.mark.parametrize("same_size", [False, True])
def test_verification_rejects_changed_content_and_truncated_zip(
    tmp_path: Path, same_size: bool
) -> None:
    run_dir = _run(tmp_path)
    destination = tmp_path / "bundle.zip"
    export_experiment_bundle(run_dir, destination)
    with zipfile.ZipFile(destination) as bundle:
        records = {name: bundle.read(name) for name in bundle.namelist()}
    records["run.json"] = b"x" * len(records["run.json"]) if same_size else b"changed"
    with zipfile.ZipFile(destination, "w") as bundle:
        for name, data in records.items():
            bundle.writestr(name, data)
    with pytest.raises(ValueError, match="integrity"):
        verify_experiment_bundle(destination)
    destination.write_bytes(destination.read_bytes()[:20])
    with pytest.raises(ValueError):
        verify_experiment_bundle(destination)


def test_write_failure_cleans_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _run(tmp_path)

    def fail(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(zipfile.ZipFile, "writestr", fail)
    with pytest.raises(OSError, match="disk full"):
        export_experiment_bundle(run_dir, tmp_path / "bundle.zip")
    assert {p.name for p in tmp_path.iterdir()} == {"run"}


def test_never_exports_datasets_secrets_or_user_code_even_when_declared(
    tmp_path: Path,
) -> None:
    run_dir = _run(
        tmp_path,
        {
            "model": "dataset/model.pt",
            "config": "credentials.json",
            "weights": "user_models/architecture.pt",
        },
    )
    for relative in (
        "dataset/model.pt",
        "credentials.json",
        "user_models/architecture.pt",
    ):
        path = run_dir / relative
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"private")
    destination = tmp_path / "bundle.zip"
    export_experiment_bundle(run_dir, destination)
    manifest = verify_experiment_bundle(destination)
    assert set(manifest["files"]) == {"run.json"}
    assert len(manifest["notes"]) == 3


def test_linked_artifact_directory_cannot_escape_run(tmp_path: Path) -> None:
    run_dir = _run(tmp_path, {"model": "linked/model.pt"})
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "model.pt").write_bytes(b"private")
    link = run_dir / "linked"
    if os.name == "nt":
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(outside)],
            check=True,
            capture_output=True,
        )
    else:
        link.symlink_to(outside, target_is_directory=True)
    destination = tmp_path / "bundle.zip"
    export_experiment_bundle(run_dir, destination)
    assert set(verify_experiment_bundle(destination)["files"]) == {"run.json"}


def test_linked_artifact_inside_run_is_also_refused(tmp_path: Path) -> None:
    run_dir = _run(tmp_path)
    real = run_dir / "run.json"
    # A symlink resolving inside the run must also be refused.
    alias = run_dir / "linked"
    if os.name == "nt":
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(alias), str(run_dir)],
            check=True,
            capture_output=True,
        )
    else:
        alias.symlink_to(run_dir, target_is_directory=True)
    (run_dir / "run.json").write_text(
        json.dumps({"artifacts": {"config": "linked/run.json"}})
    )
    destination = tmp_path / "bundle.zip"
    export_experiment_bundle(run_dir, destination)
    manifest = verify_experiment_bundle(destination)
    assert set(manifest["files"]) == {"run.json"}
    assert manifest["notes"][0]["reason"] == "outside run or linked file"
    assert real.is_file()


@pytest.mark.parametrize(
    "member",
    ["../private", "/private", "C:/private", "folder\\private", "folder/../private"],
)
def test_verify_rejects_unsafe_members(tmp_path: Path, member: str) -> None:
    destination = tmp_path / "invalid.zip"
    with zipfile.ZipFile(destination, "w") as bundle:
        bundle.writestr(member, b"private")
    if "\\" in member:
        # ZipInfo normalizes separators on Windows; corrupt the on-disk names.
        destination.write_bytes(
            destination.read_bytes().replace(b"folder/private", b"folder\\private")
        )
    with pytest.raises(ValueError, match="unsafe"):
        verify_experiment_bundle(destination)


def test_verify_rejects_duplicate_members(tmp_path: Path) -> None:
    destination = tmp_path / "invalid.zip"
    with zipfile.ZipFile(destination, "w") as bundle:
        bundle.writestr("run.json", "{}")
        with pytest.warns(UserWarning, match="Duplicate"):
            bundle.writestr("run.json", "{}")
    with pytest.raises(ValueError, match="count"):
        verify_experiment_bundle(destination)


def test_verify_rejects_symlink_members(tmp_path: Path) -> None:
    destination = tmp_path / "invalid.zip"
    info = zipfile.ZipInfo("run.json")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(destination, "w") as bundle:
        bundle.writestr(info, "../private")
    with pytest.raises(ValueError, match="linked"):
        verify_experiment_bundle(destination)


@pytest.mark.parametrize(
    "limit", ["_MAX_TOTAL_BYTES", "_MAX_MEMBERS", "_MAX_MANIFEST_BYTES"]
)
def test_verify_enforces_bounded_archive_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limit: str
) -> None:
    from visionforge.core import experiment_bundle

    run_dir = _run(tmp_path)
    destination = tmp_path / "bundle.zip"
    export_experiment_bundle(run_dir, destination)
    monkeypatch.setattr(experiment_bundle, limit, 1)
    with pytest.raises(ValueError, match="limit"):
        verify_experiment_bundle(destination)


def test_atomic_publication_does_not_replace_concurrent_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from visionforge.core import experiment_bundle

    run_dir = _run(tmp_path)
    destination = tmp_path / "bundle.zip"
    link = os.link

    def competitor(source: Path, target: Path) -> None:
        target.write_bytes(b"competitor")
        link(source, target)

    monkeypatch.setattr(experiment_bundle.os, "link", competitor)
    with pytest.raises(FileExistsError):
        export_experiment_bundle(run_dir, destination)
    assert destination.read_bytes() == b"competitor"
    assert {p.name for p in tmp_path.iterdir()} == {"run", "bundle.zip"}


@pytest.mark.parametrize(
    "manifest", [None, {"format_version": 2}, {"format_version": 1, "files": {}}]
)
def test_verify_requires_matching_supported_manifest(
    tmp_path: Path, manifest: object
) -> None:
    destination = tmp_path / "bundle.zip"
    with zipfile.ZipFile(destination, "w") as bundle:
        bundle.writestr("run.json", "{}")
        bundle.writestr("manifest.json", json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest"):
        verify_experiment_bundle(destination)


@pytest.mark.parametrize(
    "limit",
    ["_MAX_TOTAL_BYTES", "_MAX_MEMBERS", "_MAX_MANIFEST_BYTES", "_MAX_RUN_BYTES"],
)
def test_export_enforces_limits_without_partial_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limit: str
) -> None:
    from visionforge.core import experiment_bundle

    run_dir = _run(tmp_path)
    monkeypatch.setattr(experiment_bundle, limit, 1)
    with pytest.raises(ValueError, match="limit"):
        export_experiment_bundle(run_dir, tmp_path / "bundle.zip")
    assert {p.name for p in tmp_path.iterdir()} == {"run"}


def test_artifact_paths_relative_to_workspace_are_supported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _run(tmp_path, {"model": "run/weights/best.pt"})
    monkeypatch.chdir(tmp_path)
    (run_dir / "weights").mkdir()
    (run_dir / "weights/best.pt").write_bytes(b"weights")
    destination = tmp_path / "bundle.zip"
    export_experiment_bundle(run_dir, destination)
    assert "weights/best.pt" in verify_experiment_bundle(destination)["files"]


def test_source_manifest_cannot_shadow_generated_manifest(tmp_path: Path) -> None:
    run_dir = _run(tmp_path, {"config": "manifest.json"})
    (run_dir / "manifest.json").write_text('{"private": true}', encoding="utf-8")
    destination = tmp_path / "bundle.zip"
    export_experiment_bundle(run_dir, destination)
    manifest = verify_experiment_bundle(destination)
    assert set(manifest["files"]) == {"run.json"}
    assert manifest["notes"] == [
        {"artifact": "config", "reason": "reserved bundle member name"}
    ]


def _public_run(tmp_path: Path, model: str = "weights/best.pt") -> Path:
    run_dir = tmp_path / "run"
    (run_dir / "weights").mkdir(parents=True)
    (run_dir / model).write_bytes(b"weights")
    (run_dir / "curve.png").write_bytes(b"png")
    dataset = r"C:\Users\ana\Documents\VisionForge\datasets\gatos"
    record = {
        "run_dir": str(run_dir),
        "config": {"seed": 42, "data": {"base_dir": dataset}},
        "dataset_fingerprint": {"root": dataset, "sha256": "abc"},
        "history": [],
        "experiment": "gatos_001",
        "status": "completed",
        "timestamp": "2026-11-08T10:00:00",
        "metrics": {"total_epochs": 1, "accuracy": 0.9},
        "artifacts": {
            "model": str(run_dir / model),
            "graphics": [str(run_dir / "curve.png"), str(tmp_path / "outside.png")],
            "report": None,
        },
    }
    (run_dir / "run.json").write_text(json.dumps(record), encoding="utf-8")
    return run_dir


def test_a_public_bundle_has_no_local_path_and_leaves_the_run_alone(
    tmp_path: Path,
) -> None:
    run_dir = _public_run(tmp_path)
    on_disk = (run_dir / "run.json").read_bytes()
    destination = tmp_path / "public.zip"

    export_experiment_bundle(run_dir, destination, public=True)

    manifest = verify_experiment_bundle(destination)
    with zipfile.ZipFile(destination) as bundle:
        text = bundle.read("run.json").decode()
    record = json.loads(text)
    assert record["run_dir"] == "."
    # Graphics are not bundle members (ADR-135), and a weights/*.pt checkpoint
    # stays out of a public bundle (ADR-188), so the record lists neither.
    assert record["artifacts"] == {
        "model": None,
        "graphics": [],
        "report": None,
    }
    assert "weights/best.pt" not in manifest["files"]
    assert {
        "artifact": "weights/best.pt",
        "reason": "checkpoint left out of a public bundle: it may embed local paths",
    } in manifest["notes"]
    assert record["config"]["data"]["base_dir"] == "gatos"
    assert record["dataset_fingerprint"] == {"root": "gatos", "sha256": "abc"}
    assert "ana" not in text
    assert str(tmp_path) not in text
    assert {
        "artifact": "run.json",
        "reason": "local paths reduced to their final name",
    } in manifest["notes"]
    assert (run_dir / "run.json").read_bytes() == on_disk


def test_a_public_bundle_imports_with_its_model(tmp_path: Path) -> None:
    from visionforge.core.experiment_bundle import import_experiment_bundle

    destination = tmp_path / "public.zip"
    run_dir = _public_run(tmp_path, model="best_model.pth")
    export_experiment_bundle(run_dir, destination, public=True)

    imported = import_experiment_bundle(destination, tmp_path / "models")

    record = json.loads((imported / "run.json").read_text(encoding="utf-8"))
    assert Path(record["artifacts"]["model"]) == imported / "best_model.pth"
    assert (imported / "best_model.pth").read_bytes() == b"weights"


_HOME_MARKER = "vfhomemarker"


def _run_with_home_paths(tmp_path: Path) -> Path:
    """A detection-like run whose checkpoint and configs name the home folder."""
    import torch
    import yaml

    run_dir = tmp_path / "run"
    (run_dir / "weights").mkdir(parents=True)
    windows = rf"C:\Users\{_HOME_MARKER}\Documents\VisionForge\datasets\gatos"
    posix = f"/home/{_HOME_MARKER}/VisionForge/outputs/detection"
    # Ultralytics pickles its train_args, with absolute paths, into each checkpoint.
    for name in ("best.pt", "last.pt"):
        torch.save(
            {"train_args": {"data": windows, "project": posix}},
            run_dir / "weights" / name,
        )
    torch.save({"weight": torch.zeros(2)}, run_dir / "best_model.pth")
    config = {"seed": 42, "data": {"base_dir": windows}, "output": {"dir": posix}}
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    record = {
        "run_dir": str(run_dir),
        "config": config,
        "history": [],
        "artifacts": {"model": str(run_dir / "weights/best.pt")},
    }
    (run_dir / "run.json").write_text(json.dumps(record), encoding="utf-8")
    return run_dir


def test_no_member_of_a_public_bundle_names_the_home_folder(tmp_path: Path) -> None:
    destination = tmp_path / "public.zip"

    export_experiment_bundle(_run_with_home_paths(tmp_path), destination, public=True)

    verify_experiment_bundle(destination)
    marker = _HOME_MARKER.encode()
    with zipfile.ZipFile(destination) as bundle:
        leaking = [
            info.filename
            for info in bundle.infolist()
            if marker in info.filename.encode() or marker in bundle.read(info)
        ]
    assert leaking == []


def test_a_public_bundle_rewrites_configs_and_declares_what_it_left_out(
    tmp_path: Path,
) -> None:
    import yaml

    run_dir = _run_with_home_paths(tmp_path)
    on_disk = (run_dir / "config.yaml").read_bytes()
    destination = tmp_path / "public.zip"

    export_experiment_bundle(run_dir, destination, public=True)

    manifest = verify_experiment_bundle(destination)
    with zipfile.ZipFile(destination) as bundle:
        from_yaml = yaml.safe_load(bundle.read("config.yaml"))
        from_json = json.loads(bundle.read("config.json"))
        record = json.loads(bundle.read("run.json"))
    expected = {
        "seed": 42,
        "data": {"base_dir": "gatos"},
        "output": {"dir": "detection"},
    }
    assert from_yaml == expected
    assert from_json == expected
    assert record["artifacts"]["model"] is None
    assert set(manifest["files"]) == {
        "run.json",
        "config.json",
        "config.yaml",
        "best_model.pth",
    }
    omitted = {
        note["artifact"]
        for note in manifest["notes"]
        if note["reason"].startswith("checkpoint left out")
    }
    assert omitted == {"weights/best.pt", "weights/last.pt"}
    assert (run_dir / "config.yaml").read_bytes() == on_disk


def test_a_private_bundle_keeps_checkpoints_and_configs_as_they_are(
    tmp_path: Path,
) -> None:
    run_dir = _run_with_home_paths(tmp_path)
    destination = tmp_path / "private.zip"

    export_experiment_bundle(run_dir, destination)

    manifest = verify_experiment_bundle(destination)
    assert {"weights/best.pt", "weights/last.pt"} <= set(manifest["files"])
    with zipfile.ZipFile(destination) as bundle:
        config = bundle.read("config.yaml")
    assert config == (run_dir / "config.yaml").read_bytes()


@pytest.mark.parametrize(
    ("name", "content"),
    [("config.yaml", "seed: [unclosed"), ("config.json", "{not json")],
)
def test_a_public_bundle_leaves_out_an_unreadable_config(
    tmp_path: Path, name: str, content: str
) -> None:
    run_dir = _run(tmp_path)
    (run_dir / name).write_text(content, encoding="utf-8")
    destination = tmp_path / "public.zip"

    export_experiment_bundle(run_dir, destination, public=True)

    manifest = verify_experiment_bundle(destination)
    assert name not in manifest["files"]
    assert {
        "artifact": name,
        "reason": "unreadable config left out of a public bundle",
    } in manifest["notes"]
