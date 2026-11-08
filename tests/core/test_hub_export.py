"""Verify private Hub export policy with real files and an injected transport."""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from visionforge.core.hub_export import (
    HubExportError,
    PreviewMismatchError,
    PrivateRepositoryRequiredError,
    export_run,
    preview_run,
)


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    """Create a run with a checkpoint and sensitive files that must stay local."""
    root = tmp_path / "run"
    (root / "weights").mkdir(parents=True)
    (root / "weights/best.pt").write_bytes(b"opaque-checkpoint")
    (root / "logs").mkdir()
    (root / "logs/train.log").write_text("private-log-canary")
    (root / "config.yaml").write_text("token: private-token-canary")
    data = {
        "config": {
            "task": "multiclass",
            "model": {"name": "resnet18", "weights": "C:/private-weights"},
            "secret": "private-config-canary",
            "data": {"base_dir": "C:/private-dataset"},
        },
        "run_dir": str(root),
        "status": "completed",
        "metrics": {
            "accuracy": 0.875,
            "token": "private-token-canary",
            "unknown_numeric_secret": 12345,
        },
        "history": [{"epoch": 1, "val_loss": 0.1, "secret": "private-history"}],
        "artifacts": {
            "model": str(root / "weights/best.pt"),
            "log": str(root / "logs/train.log"),
        },
    }
    (root / "run.json").write_text(json.dumps(data))
    return root


def test_card_names_the_architecture_and_metrics_without_private_values(
    run_dir: Path,
) -> None:
    card = preview_run(run_dir)["card"]
    assert "Architecture: `resnet18`." in card
    assert "| accuracy | 0.875 |" in card
    assert "private" not in card
    assert "C:/" not in card
    assert "12345" not in card


def test_split_prefixed_metrics_are_exported(tmp_path: Path) -> None:
    # Classification runs record test_accuracy, test_f1 ...; detection map50.
    root = tmp_path / "run"
    root.mkdir()
    (root / "model.pt").write_bytes(b"x")
    metrics = {
        "test_accuracy": 0.9,
        "test_auc_roc": 0.95,
        "val_f1": 0.8,
        "best_auroc": 0.7,
        "test_secret_score": 1.0,
        "box_loss": 0.2,
    }
    record = {"artifacts": {"model": "model.pt"}, "metrics": metrics}
    (root / "run.json").write_text(json.dumps(record))
    card = preview_run(root)["card"]
    assert "| test_accuracy | 0.9 |" in card
    assert "| test_auc_roc | 0.95 |" in card
    assert "| val_f1 | 0.8 |" in card
    assert "| best_auroc | 0.7 |" in card
    assert "secret" not in card
    assert "box_loss" not in card


def test_card_without_architecture_or_metrics_says_nothing_about_them(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    root.mkdir()
    (root / "model.pt").write_bytes(b"x")
    record = {"config": {"task": "detection"}, "artifacts": {"model": "model.pt"}}
    (root / "run.json").write_text(json.dumps(record))
    card = preview_run(root)["card"]
    assert "Architecture" not in card
    assert "| Metric |" not in card


def _transport(
    monkeypatch: pytest.MonkeyPatch, *, private: Any = True, failure: bool = False
) -> tuple[list[Any], dict[str, bytes]]:
    calls: list[Any] = []
    uploaded: dict[str, bytes] = {}

    class Client:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(("init", kwargs))

        def create_repo(self, repo_id: str, **kwargs: Any) -> None:
            calls.append(("create", repo_id, kwargs))

        def repo_info(self, repo_id: str, **kwargs: Any) -> Any:
            calls.append(("info", repo_id, kwargs))
            return types.SimpleNamespace(private=private)

        def upload_folder(self, **kwargs: Any) -> None:
            calls.append(("upload", kwargs))
            if failure:
                raise RuntimeError("token=private-auth-canary C:/private-file")
            root = Path(kwargs["folder_path"])
            uploaded.update(
                {
                    p.relative_to(root).as_posix(): p.read_bytes()
                    for p in root.rglob("*")
                    if p.is_file()
                }
            )

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=Client)
    )
    return calls, uploaded


def test_private_export_sends_exact_confirmed_files(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = (run_dir / "run.json").read_bytes()
    monkeypatch.setenv("HF_ENDPOINT", "https://attacker.invalid")
    calls, files = _transport(monkeypatch)
    preview = preview_run(run_dir)
    assert calls == []
    assert preview == preview_run(run_dir)
    result = export_run(
        run_dir, "owner/model", "private-auth-canary", preview["digest"]
    )
    assert result == {
        "repo_id": "owner/model",
        "url": "https://huggingface.co/owner/model",
        "files": preview["files"],
    }
    assert list(files) != []
    assert set(files) == {"README.md", "run.json", "manifest.json", "weights/model.pt"}
    assert calls[0] == (
        "init",
        {"endpoint": "https://huggingface.co", "token": "private-auth-canary"},
    )
    assert calls[1] == (
        "create",
        "owner/model",
        {
            "token": "private-auth-canary",
            "private": True,
            "repo_type": "model",
            "exist_ok": True,
        },
    )
    assert calls[2][0] == "info" and calls[3][0] == "upload"
    assert calls[3][1]["allow_patterns"] == preview["files"]
    assert files["weights/model.pt"] == b"opaque-checkpoint"
    record = json.loads(files["run.json"])
    assert record == {
        "export_format": "visionforge-hub-1",
        "task": "classification",
        "status": "completed",
        "metrics": {"accuracy": 0.875},
        "history": [{"epoch": 1, "val_loss": 0.1}],
    }
    assert (run_dir / "run.json").read_bytes() == original
    manifest = json.loads(files["manifest.json"])
    for name, info in manifest.items():
        assert info == {
            "size": len(files[name]),
            "sha256": hashlib.sha256(files[name]).hexdigest(),
        }
    assert hashlib.sha256(files["manifest.json"]).hexdigest() == preview["digest"]
    for name, content in files.items():
        if name != "weights/model.pt":
            assert b"private-" not in content and b"C:/" not in content


@pytest.mark.parametrize("private", [False, None, "true", 1])
def test_unverified_privacy_stops_before_upload(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch, private: Any
) -> None:
    calls, uploaded = _transport(monkeypatch, private=private)
    with pytest.raises(PrivateRepositoryRequiredError):
        export_run(run_dir, "owner/model", "explicit", preview_run(run_dir)["digest"])
    assert [call[0] for call in calls] == ["init", "create", "info"]
    assert uploaded == {}


def test_changed_checkpoint_requires_new_confirmation(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls, _ = _transport(monkeypatch)
    digest = preview_run(run_dir)["digest"]
    (run_dir / "weights/best.pt").write_bytes(b"changed")
    with pytest.raises(PreviewMismatchError):
        export_run(run_dir, "owner/model", "explicit", digest)
    assert calls == []


def test_sdk_error_never_exposes_token_or_local_path(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _transport(monkeypatch, failure=True)
    with pytest.raises(HubExportError) as error:
        export_run(
            run_dir,
            "owner/model",
            "private-auth-canary",
            preview_run(run_dir)["digest"],
        )
    assert str(error.value) == "Hugging Face Hub export failed"
    assert error.value.__suppress_context__


def test_missing_extra_has_a_sanitised_error(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(HubExportError, match="optional"):
        export_run(run_dir, "owner/model", "explicit", preview_run(run_dir)["digest"])


@pytest.mark.parametrize(
    "artifact",
    [
        "../outside.pt",
        "logs/checkpoint.pt",
        "config.pt",
        "secrets/weights.pt",
        "weights/best.yaml",
    ],
)
def test_artifact_policy_refuses_unsafe_selection(run_dir: Path, artifact: str) -> None:
    record = json.loads((run_dir / "run.json").read_text())
    record["artifacts"]["model"] = artifact
    (run_dir / "run.json").write_text(json.dumps(record))
    with pytest.raises((ValueError, FileNotFoundError)):
        preview_run(run_dir)


def test_linked_checkpoint_directory_is_refused(run_dir: Path, tmp_path: Path) -> None:
    target = tmp_path / "external"
    target.mkdir()
    (target / "best.pt").write_bytes(b"external")
    link = run_dir / "linked"
    if sys.platform == "win32":
        import _winapi

        _winapi.CreateJunction(str(target), str(link))
    else:
        link.symlink_to(target, target_is_directory=True)
    record = json.loads((run_dir / "run.json").read_text())
    record["artifacts"]["model"] = "linked/best.pt"
    (run_dir / "run.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Linked"):
        preview_run(run_dir)


@pytest.mark.parametrize(
    "repo_id",
    [
        "model",
        "../model",
        "owner/../model",
        "owner/model.git",
        "https://attacker.invalid/model",
        "owner/a--b",
    ],
)
def test_invalid_destination_never_constructs_client(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch, repo_id: str
) -> None:
    calls, _ = _transport(monkeypatch)
    with pytest.raises(ValueError, match="owner/name"):
        export_run(run_dir, repo_id, "explicit", preview_run(run_dir)["digest"])
    assert calls == []


def test_explicit_token_required(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls, _ = _transport(monkeypatch)
    with pytest.raises(ValueError, match="explicit"):
        export_run(run_dir, "owner/model", " ", preview_run(run_dir)["digest"])
    assert calls == []


def test_record_change_requires_new_preview(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls, _ = _transport(monkeypatch)
    previous = preview_run(run_dir)["digest"]
    record = json.loads((run_dir / "run.json").read_text())
    record["metrics"]["accuracy"] = 0.5
    (run_dir / "run.json").write_text(json.dumps(record))
    with pytest.raises(PreviewMismatchError):
        export_run(run_dir, "owner/model", "explicit", previous)
    assert calls == []


@pytest.mark.parametrize("artifact", [None, "weights/best.yaml"])
def test_missing_or_unsupported_checkpoint(run_dir: Path, artifact: str | None) -> None:
    (run_dir / "weights/best.yaml").write_text("not weights")
    record = json.loads((run_dir / "run.json").read_text())
    record["artifacts"]["model"] = artifact
    (run_dir / "run.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="checkpoint"):
        preview_run(run_dir)


def test_copy_detects_source_mutation(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    original_copy = shutil.copyfileobj

    def mutate(reader: Any, writer: Any) -> None:
        original_copy(reader, writer)
        (run_dir / "weights/best.pt").write_bytes(b"source changed during copy")

    monkeypatch.setattr(shutil, "copyfileobj", mutate)
    with pytest.raises(ValueError, match="changed"):
        preview_run(run_dir)


def test_workspace_relative_checkpoint_remains_contained(
    run_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    record = json.loads((run_dir / "run.json").read_text())
    record["artifacts"]["model"] = "run/weights/best.pt"
    (run_dir / "run.json").write_text(json.dumps(record))
    assert "weights/model.pt" in preview_run(run_dir)["files"]


@pytest.mark.parametrize(
    "task", ["multiclass", "regression", "segmentation", "anomaly", "detection"]
)
def test_task_checkpoint_names_and_relative_formats(
    run_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, task: str
) -> None:
    monkeypatch.chdir(tmp_path)
    model = "weights/best.pt" if task == "detection" else "best_model.pth"
    (run_dir / model).write_bytes(b"task-checkpoint")
    record = json.loads((run_dir / "run.json").read_text())
    record["config"]["task"] = task
    previews = []
    for value in (model, str(run_dir / model), f"run/{model}"):
        record["artifacts"]["model"] = value
        (run_dir / "run.json").write_text(json.dumps(record))
        previews.append(preview_run(run_dir))
    assert previews[0] == previews[1] == previews[2]


@pytest.mark.parametrize(
    "field,value",
    [
        ("config", []),
        ("config", {"task": []}),
        ("status", {}),
        ("metrics", []),
        ("metrics", {"accuracy": []}),
        ("history", {}),
        ("history", [None, [], "invalid", {"epoch": []}]),
    ],
)
def test_json_wrong_metadata_types_never_escape_as_typeerror(
    run_dir: Path, field: str, value: Any
) -> None:
    record = json.loads((run_dir / "run.json").read_text())
    record[field] = value
    (run_dir / "run.json").write_text(json.dumps(record))
    assert isinstance(preview_run(run_dir)["digest"], str)


@pytest.mark.parametrize("value", [[], None, {"model": []}, {"model": {}}])
def test_json_wrong_artifact_types_raise_validation_error(
    run_dir: Path, value: Any
) -> None:
    record = json.loads((run_dir / "run.json").read_text())
    record["artifacts"] = value
    (run_dir / "run.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="checkpoint"):
        preview_run(run_dir)


def test_workspace_fallback_cannot_select_another_run(
    run_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    (other / "best.pt").write_bytes(b"other run")
    record = json.loads((run_dir / "run.json").read_text())
    record["artifacts"]["model"] = "other/best.pt"
    (run_dir / "run.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="outside"):
        preview_run(run_dir)
