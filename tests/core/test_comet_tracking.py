"""CometTracker (ADR-200) against the in-memory Comet fake."""

from __future__ import annotations

import json
import os
import sys
from functools import partial
from pathlib import Path
from typing import Any

import pytest

from visionforge.core.comet_tracking import (
    IMPORT_SWITCHES,
    CometPrivateProjectError,
    CometTarget,
    CometTracker,
    ensure_private_project,
)
from visionforge.core.run_trackers import export_run_record
from visionforge.utils.comet_names import is_valid_comet_name

from ..fake_comet import OS_USER, FakeComet, FakeExperiment, install_fake_comet

_TOKEN = "example-comet-key"
_CONFIG: dict[str, Any] = {
    "name": "exp",
    "task": "multiclass",
    "model": {"name": "resnet18"},
    "training": {"epochs": 2, "learning_rate": 0.001},
    "data": {"base_dir": "C:/Users/someone/private-dataset"},
    "api_token": "never-sent",
}
_FINAL = {"test_accuracy": 0.9, "test_f1": 0.8}
_PRIVACY_OFF = (
    "log_code",
    "log_graph",
    "parse_args",
    "log_git_metadata",
    "log_git_patch",
    "log_env_details",
    "log_env_gpu",
    "log_env_host",
    "log_env_cpu",
    "log_env_network",
    "log_env_disk",
    "auto_output_logging",
    "auto_param_logging",
    "auto_metric_logging",
    "auto_log_co2",
)


@pytest.fixture
def sdk(monkeypatch: pytest.MonkeyPatch) -> FakeComet:
    return install_fake_comet(monkeypatch)


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    root = tmp_path / "models" / "exp" / "20261112_120000_000000"
    root.mkdir(parents=True)
    record = {
        "config": _CONFIG,
        "metrics": {**_FINAL, "secret_score": 7},
        "history": [
            {"epoch": 1, "train_loss": 0.7, "val_loss": 0.6},
            {"epoch": 2, "train_loss": 0.5, "val_loss": 0.4},
        ],
    }
    (root / "run.json").write_text(json.dumps(record), encoding="utf-8")
    return root


def _tracker(
    run_dir: Path, *, token: str | None = _TOKEN, workspace: str = ""
) -> CometTracker:
    return CometTracker(
        "default", run_dir, CometTarget("visionforge", workspace, token)
    )


def _sidecar(run_dir: Path) -> dict[str, Any]:
    saved: dict[str, Any] = json.loads((run_dir / "comet.json").read_text("utf-8"))
    return saved


def _record(run_dir: Path) -> dict[str, Any]:
    record: dict[str, Any] = json.loads((run_dir / "run.json").read_text("utf-8"))
    return record


def _train(tracker: CometTracker, status: str = "FINISHED") -> None:
    tracker.start(_CONFIG)
    tracker.epoch(1, {"loss/train": 0.7, "loss/val": 0.6})
    tracker.epoch(2, {"loss/train": 0.5, "loss/val": float("nan")})
    tracker.finish(status)


def _only(sdk: FakeComet) -> FakeExperiment:
    (experiment,) = sdk.experiments.values()
    return experiment


def test_two_epochs_reach_a_private_project_with_public_values_only(
    run_dir: Path, sdk: FakeComet
) -> None:
    _train(_tracker(run_dir))

    assert sdk.projects == {("ana", "visionforge"): False}
    assert ("create_project", ("ana", "visionforge", False)) in sdk.calls
    (start,) = sdk.starts
    assert start["api_key"] == _TOKEN
    assert start["workspace"] == "ana"
    assert start["project_name"] == "visionforge"
    assert start["mode"] == "get_or_create"
    assert start["online"] is True
    assert len(start["experiment_key"]) == 32
    experiment = _only(sdk)
    assert experiment.parameters == {
        "model.name": "resnet18",
        "name": "exp",
        "task": "multiclass",
        "training.epochs": 2,
        "training.learning_rate": 0.001,
    }
    assert experiment.metrics == [
        ({"loss/train": 0.7, "loss/val": 0.6}, 1),
        ({"loss/train": 0.5}, 2),
        (_FINAL, None),
    ]
    assert experiment.others == {"visionforge_status": "FINISHED"}
    assert experiment.ends == 1
    saved = _sidecar(run_dir)
    assert saved["state"] == "finished"
    assert saved["status"] == "FINISHED"
    assert saved["url"] == experiment.url
    assert saved["workspace"] == "ana"
    assert saved["synced_epoch"] == 2


def test_nothing_about_the_machine_is_collected_or_sent(
    run_dir: Path, sdk: FakeComet
) -> None:
    _train(_tracker(run_dir))

    settings = sdk.starts[0]["experiment_config"].values
    for name in _PRIVACY_OFF:
        assert settings[name] is False, name
    assert settings["disabled"] is False
    assert settings["display_summary_level"] == 0
    assert settings["name"] == run_dir.name
    for name, value in IMPORT_SWITCHES.items():
        assert os.environ[name] == value
    experiment = _only(sdk)
    # The real end() mails a summary titled with the OS user name.
    assert experiment.notifications == []
    sent = json.dumps(
        [experiment.parameters, experiment.metrics, experiment.others, sdk.calls[1:]]
    )
    for private in (_TOKEN, "private-dataset", "never-sent", str(run_dir), OS_USER):
        assert private not in sent


def test_the_key_goes_only_to_the_sdk_never_to_the_sidecar(
    run_dir: Path, sdk: FakeComet
) -> None:
    _train(_tracker(run_dir))

    assert sdk.calls[0] == ("API", {"api_key": _TOKEN, "cache": False})
    assert _TOKEN not in (run_dir / "comet.json").read_text("utf-8")


def test_an_existing_public_project_is_refused_not_changed(
    run_dir: Path, sdk: FakeComet
) -> None:
    sdk.projects[("ana", "visionforge")] = True

    _train(_tracker(run_dir))

    assert sdk.starts == []
    assert sdk.projects == {("ana", "visionforge"): True}
    saved = _sidecar(run_dir)
    assert saved["state"] == "failed"
    assert saved["error"] == "CometPrivateProjectError"
    assert saved["status"] == "FINISHED"


def test_a_created_project_the_server_did_not_make_private_is_refused(
    sdk: FakeComet,
) -> None:
    sdk.create_public = True

    with pytest.raises(CometPrivateProjectError):
        ensure_private_project(sdk, "ana", "visionforge")


def test_an_existing_private_project_is_used_as_is(sdk: FakeComet) -> None:
    sdk.projects[("lab", "visionforge")] = False

    ensure_private_project(sdk, "lab", "visionforge")

    assert not any(name == "create_project" for name, _ in sdk.calls)


def test_an_explicit_workspace_skips_the_default_workspace_query(
    run_dir: Path, sdk: FakeComet
) -> None:
    _train(_tracker(run_dir, workspace="lab"))

    assert not any(name == "get_default_workspace" for name, _ in sdk.calls)
    assert sdk.starts[0]["workspace"] == "lab"
    assert sdk.projects == {("lab", "visionforge"): False}


def test_without_a_key_nothing_is_contacted(run_dir: Path, sdk: FakeComet) -> None:
    _train(_tracker(run_dir, token=None))

    assert sdk.calls == []
    assert sdk.starts == []
    assert _sidecar(run_dir)["error"] == "CometTokenMissingError"


def test_without_the_extra_the_export_is_disabled(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in IMPORT_SWITCHES:
        monkeypatch.setenv(name, "")
    monkeypatch.setitem(sys.modules, "comet_ml", None)

    _train(_tracker(run_dir))

    saved = _sidecar(run_dir)
    assert saved["state"] == "disabled"
    assert saved["error"] == "ModuleNotFoundError"


def test_a_failing_sdk_never_reaches_the_training(
    run_dir: Path, sdk: FakeComet
) -> None:
    sdk.fail_start = RuntimeError("network down")

    _train(_tracker(run_dir))

    saved = _sidecar(run_dir)
    assert saved["state"] == "failed"
    assert saved["error"] == "RuntimeError"


def test_an_experiment_that_did_not_start_is_ended_and_reported(
    run_dir: Path, sdk: FakeComet
) -> None:
    sdk.start_dead = True

    _train(_tracker(run_dir))

    experiment = _only(sdk)
    assert experiment.metrics == []
    assert experiment.ends == 1
    assert _sidecar(run_dir)["error"] == "CometRequestError"


def test_a_stopped_run_is_marked_killed(run_dir: Path, sdk: FakeComet) -> None:
    _train(_tracker(run_dir), status="KILLED")

    assert _only(sdk).others == {"visionforge_status": "KILLED"}
    assert _sidecar(run_dir)["status"] == "KILLED"


def test_a_retry_reopens_the_same_experiment_and_adds_only_missing_epochs(
    run_dir: Path, sdk: FakeComet
) -> None:
    first = _tracker(run_dir)
    first.start(_CONFIG)
    first.epoch(1, {"loss/train": 0.7})
    first.finish("FAILED")

    _tracker(run_dir).retry(_record(run_dir))

    assert sdk.starts[0]["experiment_key"] == sdk.starts[1]["experiment_key"]
    experiment = _only(sdk)
    assert experiment.steps() == [1, None, 2, None]
    assert experiment.metrics[2] == ({"loss/train": 0.5, "loss/val": 0.4}, 2)
    assert experiment.ends == 2
    saved = _sidecar(run_dir)
    assert saved["state"] == "finished"
    assert saved["status"] == "FAILED"
    assert saved["synced_epoch"] == 2


def test_an_unconfirmed_upload_is_sent_again_on_retry(
    run_dir: Path, sdk: FakeComet
) -> None:
    tracker = _tracker(run_dir)
    tracker.start(_CONFIG)
    _only(sdk).flush_ok = False
    tracker.epoch(1, {"loss/train": 0.7})
    tracker.finish("FINISHED")
    saved = _sidecar(run_dir)
    assert saved["state"] == "failed"
    assert saved["synced_epoch"] == 0

    _only(sdk).flush_ok = True
    _tracker(run_dir).retry(_record(run_dir))

    assert _only(sdk).steps() == [1, None, 1, 2, None]
    assert _sidecar(run_dir)["synced_epoch"] == 2


def test_a_second_export_waits_while_one_is_open(
    tmp_path: Path, run_dir: Path, sdk: FakeComet
) -> None:
    other = tmp_path / "models" / "exp" / "20261112_130000_000000"
    other.mkdir()
    live = _tracker(run_dir)
    live.start(_CONFIG)

    _tracker(other).start(_CONFIG)

    assert len(sdk.starts) == 1
    assert _sidecar(other)["error"] == "CometBusyError"
    live.finish("FINISHED")
    later = _tracker(other)
    later.start(_CONFIG)
    later.finish("FINISHED")
    assert len(sdk.starts) == 2
    assert _sidecar(other)["state"] == "finished"


def test_a_sidecar_from_another_run_is_refused(run_dir: Path, sdk: FakeComet) -> None:
    (run_dir / "comet.json").write_text(
        json.dumps({"identity": "someone-else", "status": "FINISHED"}),
        encoding="utf-8",
    )

    _tracker(run_dir).start(_CONFIG)

    assert sdk.starts == []
    assert _sidecar(run_dir)["state"] == "failed"


def test_a_regression_run_sends_its_history_when_it_ends(
    tmp_path: Path, sdk: FakeComet
) -> None:
    run_dir = tmp_path / "models" / "reg" / "20261112_140000_000000"
    run_dir.mkdir(parents=True)
    record = {
        "config": {"name": "reg", "data": {"base_dir": "C:/Users/someone/data"}},
        "metrics": {"mse": 0.2, "r2": 0.8},
        "history": [
            {"epoch": 1, "train_loss": 0.5, "val_loss": 0.4, "val_r2": 0.6},
            {"epoch": 2, "train_loss": 0.3, "val_loss": 0.2, "val_r2": 0.8},
        ],
    }
    (run_dir / "run.json").write_text(json.dumps(record), encoding="utf-8")
    target = CometTarget("visionforge", token=_TOKEN)

    export_run_record([partial(CometTracker, "ana", target=target)], run_dir)

    experiment = _only(sdk)
    assert experiment.parameters == {"name": "reg"}
    assert experiment.metrics == [
        ({"loss/train": 0.5, "loss/val": 0.4, "val_r2": 0.6}, 1),
        ({"loss/train": 0.3, "loss/val": 0.2, "val_r2": 0.8}, 2),
        ({"mse": 0.2, "r2": 0.8}, None),
    ]
    assert _sidecar(run_dir)["state"] == "finished"


@pytest.mark.parametrize("name", ["visionforge", "my-runs_2.0", "a" * 100, "lab"])
def test_valid_names(name: str) -> None:
    assert is_valid_comet_name(name)


@pytest.mark.parametrize(
    "name", ["", ".", "..", "-x", "a/b", "a b", "a" * 101, "Lab", "ação", "../x"]
)
def test_invalid_names(name: str) -> None:
    assert not is_valid_comet_name(name)
