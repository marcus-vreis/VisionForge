"""Importing an experiment bundle as a local run (ADR-163, phase 3)."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

from visionforge.core import experiment_bundle as eb
from visionforge.core.experiment_bundle import (
    export_experiment_bundle,
    import_experiment_bundle,
)
from visionforge.gui.api.run_history import load_runs

REMOTE_RUN = "/content/vf/outputs/models/exp/20261105_120000_000001"
CHECKPOINT = b"weights" * 64


def _record(**overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": "exp_20261105_120000_000001",
        "experiment": "exp",
        "timestamp": "2026-11-05T12:01:00",
        "status": "completed",
        "stopped": False,
        "device_used": "cuda",
        "environment": {"python": "3.12.12", "torch": "2.9.0+cu126", "gpu": "Tesla T4"},
        "run_dir": REMOTE_RUN,
        "config": {"name": "exp", "schema_version": 1, "model": {"name": "resnet18"}},
        "metrics": {"best_val_loss": 0.4, "best_epoch": 2, "total_epochs": 2},
        "history": [
            {"epoch": 1, "val_loss": 0.6, "val_accuracy": 0.68},
            {"epoch": 2, "val_loss": 0.4, "val_accuracy": 0.82},
        ],
        "artifacts": {
            "model": f"{REMOTE_RUN}/best_model.pth",
            "graphics": [f"{REMOTE_RUN}/confusion_matrix.png"],
            "report": None,
        },
    }
    record.update(overrides)
    return record


def _bundle(path: Path, record: dict[str, Any], members: dict[str, bytes]) -> Path:
    """A bundle as the VM's export writes it: run.json, members, manifest."""
    raw = json.dumps(record).encode()
    files = {"run.json": raw, **members}
    manifest = {
        "format_version": 1,
        "files": {
            name: {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            for name, data in files.items()
        },
        "notes": [],
    }
    with zipfile.ZipFile(path, "w") as bundle:
        for name, data in files.items():
            bundle.writestr(name, data)
        bundle.writestr("manifest.json", json.dumps(manifest))
    return path


def _remote_bundle(tmp_path: Path, **overrides: Any) -> Path:
    return _bundle(
        tmp_path / "run-bundle.zip",
        _record(**overrides),
        {"best_model.pth": CHECKPOINT},
    )


def test_import_places_the_run_under_its_experiment_and_rebases_its_paths(
    tmp_path: Path,
) -> None:
    models = tmp_path / "outputs" / "models"
    origin = {"kind": "colab", "accelerator": "T4"}

    run_dir = import_experiment_bundle(_remote_bundle(tmp_path), models, origin=origin)

    assert run_dir == models.resolve() / "exp" / "20261105_120000_000001"
    assert (run_dir / "best_model.pth").read_bytes() == CHECKPOINT
    data = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert data["run_dir"] == str(run_dir)
    assert data["artifacts"]["model"] == str(run_dir / "best_model.pth")
    # The bundle format carries no plots: nothing may point at the VM's folder.
    assert data["artifacts"]["graphics"] == []
    assert data["origin"] == origin
    # What the VM recorded about itself is kept as it was.
    assert data["device_used"] == "cuda"
    assert data["environment"]["gpu"] == "Tesla T4"
    assert sorted(p.name for p in models.parent.iterdir()) == ["models"]


def test_an_imported_run_is_listed_by_history_with_its_origin(tmp_path: Path) -> None:
    models = tmp_path / "models"
    import_experiment_bundle(_remote_bundle(tmp_path), models, origin={"kind": "colab"})

    [summary] = load_runs(models)
    assert summary.run_id == "20261105_120000_000001"
    assert summary.experiment_name == "exp"
    assert summary.origin == "colab"
    assert summary.epochs_completed == 2


def test_a_bundle_exported_here_imports_the_same_way(tmp_path: Path) -> None:
    """The import a Colab job uses also takes a bundle downloaded by hand (plan F1)."""
    source = tmp_path / "elsewhere" / "exp" / "20261101_090000_000000"
    source.mkdir(parents=True)
    (source / "best_model.pth").write_bytes(CHECKPOINT)
    record = _record(
        run_dir=str(source),
        artifacts={"model": str(source / "best_model.pth"), "graphics": []},
    )
    (source / "run.json").write_text(json.dumps(record), encoding="utf-8")
    bundle = export_experiment_bundle(source, tmp_path / "manual.zip")

    run_dir = import_experiment_bundle(bundle, tmp_path / "models")

    assert run_dir.name == "20261101_090000_000000"
    assert (run_dir / "best_model.pth").read_bytes() == CHECKPOINT
    data = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert "origin" not in data
    # The path recorded on this OS (a drive path on Windows) is rebased too.
    assert data["artifacts"]["model"] == str(run_dir / "best_model.pth")
    assert [s.origin for s in load_runs(tmp_path / "models")] == [None]


def test_a_windows_run_path_is_rebased_on_any_os(tmp_path: Path) -> None:
    origin = r"C:\Users\ana\outputs\models\exp\20261101_090000_000000"
    record = _record(
        run_dir=origin,
        artifacts={"model": origin + r"\best_model.pth", "graphics": []},
    )
    bundle = _bundle(tmp_path / "win.zip", record, {"best_model.pth": CHECKPOINT})

    run_dir = import_experiment_bundle(bundle, tmp_path / "models")

    assert run_dir.name == "20261101_090000_000000"
    data = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert data["artifacts"]["model"] == str(run_dir / "best_model.pth")


def _assert_nothing_written(models: Path) -> None:
    assert not models.exists() or list(models.rglob("*")) == []
    leftovers = [p for p in models.parent.iterdir() if p.name.startswith(".")]
    assert leftovers == []


@pytest.mark.parametrize("member", ["../escape.pth", "/abs.pth", "a/../../b.pth"])
def test_a_zip_slip_member_is_refused_before_anything_is_written(
    tmp_path: Path, member: str
) -> None:
    models = tmp_path / "models"
    bundle = _bundle(tmp_path / "evil.zip", _record(), {member: b"x"})

    with pytest.raises(ValueError, match="unsafe"):
        import_experiment_bundle(bundle, models)

    _assert_nothing_written(models)
    assert not (tmp_path / "escape.pth").exists()


@pytest.mark.parametrize("experiment", ["../up", ".hidden", "a/b", ""])
def test_an_experiment_name_that_is_not_one_folder_is_refused(
    tmp_path: Path, experiment: str
) -> None:
    models = tmp_path / "models"
    bundle = _remote_bundle(tmp_path, experiment=experiment)
    with pytest.raises(ValueError, match="complete run record"):
        import_experiment_bundle(bundle, models)
    _assert_nothing_written(models)


def test_a_run_folder_name_that_would_escape_gets_a_fresh_stamp(
    tmp_path: Path,
) -> None:
    run_dir = import_experiment_bundle(
        _remote_bundle(tmp_path, run_dir="/content/vf/.."), tmp_path / "models"
    )
    assert run_dir.parent == (tmp_path / "models").resolve() / "exp"
    assert run_dir.name[:8].isdigit()


def test_an_archive_over_the_limit_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _remote_bundle(tmp_path)
    monkeypatch.setattr(eb, "_MAX_IMPORT_BYTES", bundle.stat().st_size - 1)
    with pytest.raises(ValueError, match="size limit"):
        import_experiment_bundle(bundle, tmp_path / "models")
    _assert_nothing_written(tmp_path / "models")


def test_contents_over_the_limit_are_refused_and_cleaned_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A highly compressible member (a zip bomb) is caught by what it unpacks to."""
    models = tmp_path / "models"
    bundle = _bundle(tmp_path / "bomb.zip", _record(), {"best_model.pth": b"0" * 4096})
    monkeypatch.setattr(eb, "_MAX_IMPORT_BYTES", 4000)
    with pytest.raises(ValueError, match="size limit"):
        import_experiment_bundle(bundle, models)
    _assert_nothing_written(models)


def test_a_failure_while_unpacking_leaves_no_staging_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models = tmp_path / "models"

    def broken(*args: object, **kwargs: object) -> set[str]:
        raise OSError("disk full")

    monkeypatch.setattr(eb, "_extract_members", broken)
    bundle = _remote_bundle(tmp_path)
    with pytest.raises(OSError, match="disk full"):
        import_experiment_bundle(bundle, models)
    _assert_nothing_written(models)


def test_an_existing_run_is_never_replaced(tmp_path: Path) -> None:
    models = tmp_path / "outputs" / "models"
    first = import_experiment_bundle(_remote_bundle(tmp_path), models)
    (first / "note.txt").write_text("mine", encoding="utf-8")

    bundle = _remote_bundle(tmp_path)
    with pytest.raises(FileExistsError):
        import_experiment_bundle(bundle, models)

    assert (first / "note.txt").read_text(encoding="utf-8") == "mine"
    assert [p.name for p in models.parent.iterdir()] == ["models"]


def test_a_run_from_a_newer_schema_is_refused(tmp_path: Path) -> None:
    config = {"name": "exp", "schema_version": 999}
    bundle = _remote_bundle(tmp_path, config=config)
    with pytest.raises(ValueError):
        import_experiment_bundle(bundle, tmp_path / "models")
    _assert_nothing_written(tmp_path / "models")


def test_a_record_without_its_epoch_count_is_refused(tmp_path: Path) -> None:
    bundle = _remote_bundle(tmp_path, metrics={})
    with pytest.raises(ValueError, match="complete run record"):
        import_experiment_bundle(bundle, tmp_path / "models")


def test_a_changed_member_fails_the_checksum(tmp_path: Path) -> None:
    bundle = _remote_bundle(tmp_path)
    tampered = bundle.read_bytes().replace(CHECKPOINT[:7], b"WEIGHTS")
    bundle.write_bytes(tampered)
    with pytest.raises(ValueError):
        import_experiment_bundle(bundle, tmp_path / "models")
    _assert_nothing_written(tmp_path / "models")
