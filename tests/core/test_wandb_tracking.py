"""WandbTracker (ADR-172) against the in-memory W&B fakes."""

from __future__ import annotations

import base64
import json
import sys
import urllib.error
from pathlib import Path
from typing import Any

import pytest

from visionforge.core import wandb_tracking
from visionforge.core.wandb_tracking import (
    WandbPrivateProjectError,
    WandbRequestError,
    WandbTarget,
    WandbTracker,
    ensure_private_project,
    http_graphql,
    is_valid_wandb_name,
)

from ..fake_wandb import FakeGraphQL, FakeWandb, install_fake_wandb

_TOKEN = "example-wandb-key"
_CONFIG: dict[str, Any] = {
    "name": "exp",
    "task": "multiclass",
    "model": {"name": "resnet18"},
    "training": {"epochs": 2, "learning_rate": 0.001},
    "data": {"base_dir": "C:/Users/someone/private-dataset"},
    "api_token": "never-sent",
}


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch) -> tuple[FakeWandb, FakeGraphQL]:
    return install_fake_wandb(monkeypatch)


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    root = tmp_path / "models" / "exp" / "20261108_120000_000000"
    root.mkdir(parents=True)
    record = {
        "config": _CONFIG,
        "metrics": {"test_accuracy": 0.9, "test_f1": 0.8, "secret_score": 7},
        "history": [
            {"epoch": 1, "train_loss": 0.7, "val_loss": 0.6},
            {"epoch": 2, "train_loss": 0.5, "val_loss": 0.4},
        ],
    }
    (root / "run.json").write_text(json.dumps(record), encoding="utf-8")
    return root


def _tracker(
    tmp_path: Path, run_dir: Path, *, token: str | None = _TOKEN, entity: str = ""
) -> WandbTracker:
    target = WandbTarget("visionforge", entity, token)
    return WandbTracker(tmp_path / "wandb", "default", run_dir, target)


def _sidecar(run_dir: Path) -> dict[str, Any]:
    saved: dict[str, Any] = json.loads((run_dir / "wandb.json").read_text("utf-8"))
    return saved


def _train(tracker: WandbTracker, status: str = "FINISHED") -> None:
    tracker.start(_CONFIG)
    tracker.epoch(1, {"loss/train": 0.7, "loss/val": 0.6})
    tracker.epoch(2, {"loss/train": 0.5, "loss/val": float("nan")})
    tracker.finish(status)


def test_a_training_reaches_a_private_project_with_public_values_only(
    tmp_path: Path, run_dir: Path, fakes: tuple[FakeWandb, FakeGraphQL]
) -> None:
    sdk, api = fakes
    _train(_tracker(tmp_path, run_dir))

    assert api.projects == {("ana", "visionforge"): "PRIVATE"}
    (init,) = sdk.inits
    assert init["entity"] == "ana"
    assert init["project"] == "visionforge"
    assert init["config"] == {
        "model.name": "resnet18",
        "name": "exp",
        "task": "multiclass",
        "training.epochs": 2,
        "training.learning_rate": 0.001,
    }
    run = sdk.runs[init["id"]]
    assert run.history == {
        1: {"loss/train": 0.7, "loss/val": 0.6},
        2: {"loss/train": 0.5},
    }
    assert run.summary == {"test_accuracy": 0.9, "test_f1": 0.8}
    assert run.exit_codes == [0]
    saved = _sidecar(run_dir)
    assert saved["state"] == "finished"
    assert saved["status"] == "FINISHED"
    assert saved["url"] == run.url
    assert saved["entity"] == "ana"


def test_nothing_about_the_machine_is_sent(
    tmp_path: Path, run_dir: Path, fakes: tuple[FakeWandb, FakeGraphQL]
) -> None:
    sdk, _ = fakes
    _tracker(tmp_path, run_dir).start(_CONFIG)

    settings = sdk.inits[0]["settings"].values
    assert settings["api_key"] == _TOKEN
    assert settings["mode"] == "online"
    assert settings["console"] == "off"
    for flag in (
        "disable_git",
        "disable_code",
        "x_disable_meta",
        "x_disable_machine_info",
        "x_disable_stats",
    ):
        assert settings[flag] is True, flag
    assert settings["save_code"] is False
    assert settings["x_save_requirements"] is False
    assert "private-dataset" not in json.dumps(sdk.inits[0]["config"])


def test_an_existing_project_that_is_not_private_is_refused_not_changed(
    tmp_path: Path, run_dir: Path, fakes: tuple[FakeWandb, FakeGraphQL]
) -> None:
    sdk, api = fakes
    api.projects[("ana", "visionforge")] = "USER_READ"

    _train(_tracker(tmp_path, run_dir))

    assert sdk.inits == []
    assert api.projects == {("ana", "visionforge"): "USER_READ"}
    saved = _sidecar(run_dir)
    assert saved["state"] == "failed"
    assert saved["error"] == "WandbPrivateProjectError"
    assert saved["status"] == "FINISHED"


def test_a_created_project_the_server_did_not_make_private_is_refused(
    fakes: tuple[FakeWandb, FakeGraphQL],
) -> None:
    _, api = fakes
    api.create_access = "ENTITY_READ"

    with pytest.raises(WandbPrivateProjectError):
        ensure_private_project(api, "ana", "visionforge")


def test_an_existing_private_project_is_used_as_is(
    fakes: tuple[FakeWandb, FakeGraphQL],
) -> None:
    _, api = fakes
    api.projects[("lab", "visionforge")] = "PRIVATE"

    ensure_private_project(api, "lab", "visionforge")

    assert not any("upsertModel" in query for query, _ in api.calls)


def test_an_explicit_entity_skips_the_viewer_query(
    tmp_path: Path, run_dir: Path, fakes: tuple[FakeWandb, FakeGraphQL]
) -> None:
    sdk, api = fakes
    _tracker(tmp_path, run_dir, entity="lab").start(_CONFIG)

    assert not any("viewer" in query for query, _ in api.calls)
    assert sdk.inits[0]["entity"] == "lab"
    assert api.projects == {("lab", "visionforge"): "PRIVATE"}


def test_without_a_token_nothing_is_contacted(
    tmp_path: Path, run_dir: Path, fakes: tuple[FakeWandb, FakeGraphQL]
) -> None:
    sdk, api = fakes
    _train(_tracker(tmp_path, run_dir, token=None))

    assert api.calls == []
    assert sdk.inits == []
    assert _sidecar(run_dir)["error"] == "WandbTokenMissingError"


def test_without_the_extra_the_export_is_disabled(
    tmp_path: Path, run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "wandb", None)

    _train(_tracker(tmp_path, run_dir))

    saved = _sidecar(run_dir)
    assert saved["state"] == "disabled"
    assert saved["error"] == "ModuleNotFoundError"


def test_a_failing_sdk_never_reaches_the_training(
    tmp_path: Path, run_dir: Path, fakes: tuple[FakeWandb, FakeGraphQL]
) -> None:
    sdk, _ = fakes
    sdk.fail_init = RuntimeError("network down")

    _train(_tracker(tmp_path, run_dir))

    saved = _sidecar(run_dir)
    assert saved["state"] == "failed"
    assert saved["error"] == "RuntimeError"


def test_a_stopped_run_closes_with_a_non_zero_exit_code(
    tmp_path: Path, run_dir: Path, fakes: tuple[FakeWandb, FakeGraphQL]
) -> None:
    sdk, _ = fakes
    _train(_tracker(tmp_path, run_dir), status="KILLED")

    (run,) = sdk.runs.values()
    assert run.exit_codes == [1]
    assert _sidecar(run_dir)["status"] == "KILLED"


def test_a_retry_resumes_the_same_run_and_fills_the_missing_epochs(
    tmp_path: Path, run_dir: Path, fakes: tuple[FakeWandb, FakeGraphQL]
) -> None:
    sdk, _ = fakes
    first = _tracker(tmp_path, run_dir)
    first.start(_CONFIG)
    first.epoch(1, {"loss/train": 0.7})
    first.finish("FAILED")

    record = json.loads((run_dir / "run.json").read_text("utf-8"))
    _tracker(tmp_path, run_dir).retry(record)

    assert sdk.inits[0]["id"] == sdk.inits[1]["id"]
    assert sdk.inits[1]["resume"] == "allow"
    (run,) = sdk.runs.values()
    assert run.history == {
        1: {"loss/train": 0.7},
        2: {"loss/train": 0.5, "loss/val": 0.4},
    }
    assert run.exit_codes == [1, 1]
    saved = _sidecar(run_dir)
    assert saved["state"] == "finished"
    assert saved["status"] == "FAILED"


def test_a_sidecar_from_another_run_is_refused(
    tmp_path: Path, run_dir: Path, fakes: tuple[FakeWandb, FakeGraphQL]
) -> None:
    sdk, _ = fakes
    (run_dir / "wandb.json").write_text(
        json.dumps({"identity": "someone-else", "status": "FINISHED"}),
        encoding="utf-8",
    )

    _tracker(tmp_path, run_dir).start(_CONFIG)

    assert sdk.inits == []
    assert _sidecar(run_dir)["state"] == "failed"


@pytest.mark.parametrize("name", ["visionforge", "my-runs_2.0", "a" * 128, "Lab.Team"])
def test_valid_names(name: str) -> None:
    assert is_valid_wandb_name(name)


@pytest.mark.parametrize(
    "name", ["", ".", "..", "a/b", "a b", "a" * 129, "ação", "../x"]
)
def test_invalid_names(name: str) -> None:
    assert not is_valid_wandb_name(name)


class _Response:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode()

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        return None


def test_graphql_sends_only_this_token_over_https(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[Any] = []

    def urlopen(request: Any, timeout: int) -> _Response:
        seen.append(request)
        return _Response({"data": {"viewer": {"entity": "ana"}}})

    monkeypatch.setattr(wandb_tracking.urllib.request, "urlopen", urlopen)
    call = http_graphql("https://api.example/", _TOKEN)

    assert call("query { viewer { entity } }", {}) == {"viewer": {"entity": "ana"}}
    (request,) = seen
    assert request.full_url == "https://api.example/graphql"
    expected = base64.b64encode(f"api:{_TOKEN}".encode()).decode()
    assert request.get_header("Authorization") == f"Basic {expected}"


def test_graphql_refuses_plain_http() -> None:
    with pytest.raises(WandbRequestError):
        http_graphql("http://api.example", _TOKEN)


@pytest.mark.parametrize(
    "outcome",
    [
        urllib.error.URLError(f"refused for {_TOKEN}"),
        _Response({"errors": [{"message": f"bad key {_TOKEN}"}]}),
    ],
)
def test_graphql_errors_never_carry_the_token(
    monkeypatch: pytest.MonkeyPatch, outcome: Any
) -> None:
    def urlopen(request: Any, timeout: int) -> _Response:
        if isinstance(outcome, Exception):
            raise outcome
        response: _Response = outcome
        return response

    monkeypatch.setattr(wandb_tracking.urllib.request, "urlopen", urlopen)

    graphql = http_graphql("https://api.example", _TOKEN)
    with pytest.raises(WandbRequestError) as caught:
        graphql("query { viewer { entity } }", {})
    assert _TOKEN not in str(caught.value)
    assert caught.value.__cause__ is None
