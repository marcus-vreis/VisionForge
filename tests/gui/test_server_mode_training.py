"""Server mode: a training config names only what its owner may reach (ADR-215).

The configs are shared with the command line, so the rule is applied to the
validated config by the run routes: paths inside the person's folder or a
shared one, names inside the dataset that only go down, a model name that is
not a path, and no custom model.
"""

from __future__ import annotations

import typing
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel

from visionforge.gui.api import routes, submission, training_routes
from visionforge.gui.api.config_scope import (
    DATASET_PARTS,
    FORBIDDEN,
    FREE_TEXT,
    MODEL_NAMES,
    OPEN_BLOCKS,
    REWRITING,
    closed_blocks,
    confine_config,
)
from visionforge.gui.api.dataset_scope import DatasetScope, dataset_scope
from visionforge.gui.api.profiles import Profile, ensure_profile
from visionforge.gui.server_mode import ServerMode, user_slug
from visionforge.utils.anomaly_config import AnomalyConfig
from visionforge.utils.config import ExperimentConfig
from visionforge.utils.detection_config import DetectionConfig
from visionforge.utils.regression_config import RegressionConfig
from visionforge.utils.segmentation_config import SegmentationConfig

PROXY = "10.0.0.5"
ANA = {"X-Forwarded-User": "ana@lab.org"}
CONFIGS = (
    ExperimentConfig,
    DetectionConfig,
    RegressionConfig,
    SegmentationConfig,
    AnomalyConfig,
)


def _text_fields(model: type[BaseModel], prefix: str = "") -> Iterator[str]:
    """Dotted names of the fields of ``model`` that may hold text."""
    for name, field in model.model_fields.items():
        pending = [field.annotation]
        while pending:
            kind = pending.pop()
            if typing.get_origin(kind) is not None:
                pending.extend(typing.get_args(kind))
            elif isinstance(kind, type) and issubclass(kind, BaseModel):
                yield from _text_fields(kind, f"{prefix}{name}.")
            elif kind is str:
                yield f"{prefix}{name}"


def test_every_text_field_of_a_config_was_read() -> None:
    known = MODEL_NAMES | FORBIDDEN | FREE_TEXT | REWRITING
    unread = {
        f"{config.__name__}: {field}"
        for config in CONFIGS
        for field in _text_fields(config)
        if field not in known and field.rsplit(".", 1)[-1] not in DATASET_PARTS
    }
    assert unread == set(), "text fields of a config with no server-mode rule"


@pytest.fixture
def world(tmp_path: Path) -> dict[str, Any]:
    """A shared dataset, ana's profile, and a folder nobody may reach."""
    shared = tmp_path / "shared"
    (shared / "beans" / "train" / "healthy").mkdir(parents=True)
    secret = tmp_path / "secret"
    secret.mkdir()
    (secret / "weights.pt").write_bytes(b"not a checkpoint")
    profile = ensure_profile(user_slug("ana@lab.org"), "ana", tmp_path / "profiles")
    own = (tmp_path / "profiles" / profile.slug).resolve()
    (own / "datasets").mkdir()
    (own / "models" / "last.pth").write_bytes(b"mine")
    return {
        "shared": shared.resolve(),
        "secret": secret.resolve(),
        "profile": profile,
        "own": own,
        "scope": DatasetScope(
            readable=(own / "datasets", shared.resolve()), owner=profile.slug
        ),
    }


def _classification(base_dir: Path, **over: Any) -> ExperimentConfig:
    body: dict[str, Any] = {
        "name": "beans",
        "model": {"name": "resnet18", "num_classes": 2, "pretrained": False},
        "data": {"base_dir": str(base_dir)},
    }
    for key, value in over.items():
        body[key] = {**body.get(key, {}), **value}
    return ExperimentConfig.model_validate(body)


def _checked(config: BaseModel, world: dict[str, Any]) -> BaseModel:
    profile: Profile = world["profile"]
    scoped = profile.scope_model(config)
    with dataset_scope(world["scope"]):
        confine_config(scoped, profile)
    return scoped


def _refused(config: BaseModel, world: dict[str, Any]) -> str:
    with pytest.raises(HTTPException) as refusal:
        _checked(config, world)
    assert refusal.value.status_code == 422
    return str(refusal.value.detail)


def test_on_a_local_install_a_config_may_name_any_folder(world: dict[str, Any]) -> None:
    config = _classification(world["secret"], data={"train_dir": "../elsewhere"})
    confine_config(config, world["profile"])


def test_a_shared_dataset_and_the_persons_own_checkpoint_are_accepted(
    world: dict[str, Any],
) -> None:
    config = _classification(
        world["shared"] / "beans",
        model={"weights_path": str(world["own"] / "models" / "last.pth")},
    )
    scoped = _checked(config, world)
    assert Path(scoped.output.models_dir).resolve().is_relative_to(world["own"])  # type: ignore[attr-defined]


def test_a_dataset_outside_the_scope_is_refused(world: dict[str, Any]) -> None:
    assert "data.base_dir" in _refused(_classification(world["secret"]), world)
    other = world["own"].parent / "someone-else"
    other.mkdir()
    assert "data.base_dir" in _refused(_classification(other), world)


def test_a_split_name_cannot_climb_out_of_the_dataset(world: dict[str, Any]) -> None:
    config = _classification(
        world["shared"] / "beans", data={"train_dir": "../../secret"}
    )
    assert "../../secret" in _refused(config, world)


def test_weights_outside_the_scope_are_refused(world: dict[str, Any]) -> None:
    config = _classification(
        world["shared"] / "beans",
        model={"weights_path": str(world["secret"] / "weights.pt")},
    )
    assert "model.weights_path" in _refused(config, world)


@pytest.mark.parametrize(
    ("axis", "target"),
    [
        ("data.base_dir", "secret"),
        ("model.weights_path", "secret/weights.pt"),
        ("output.models_dir", "secret"),
        ("model.custom_model", "mine"),
    ],
)
def test_a_search_axis_cannot_name_what_the_config_may_not(
    world: dict[str, Any], axis: str, target: str
) -> None:
    """A grid axis rewrites a field of the config for each trial, after the
    config was confined: the search is refused whole until its axes are read."""
    value = (
        str(world["secret"].parent / target)
        if "/" in target or target == "secret"
        else target
    )
    config = _classification(
        world["shared"] / "beans", grid_search={"hyperparameters": {axis: [value]}}
    ).model_copy(update={"block": "grid_search"})
    assert "grid_search" in _refused(config, world)
    confine_config(config, world["profile"])  # the local tool keeps its search


@pytest.mark.parametrize("block", sorted(closed_blocks()))
def test_only_the_plain_run_of_classification_is_open(
    world: dict[str, Any], block: str
) -> None:
    config = _classification(world["shared"] / "beans").model_copy(
        update={"block": block}
    )
    assert block in _refused(config, world)


def test_the_closed_blocks_are_every_block_but_the_open_ones() -> None:
    every = set(typing.get_args(ExperimentConfig.model_fields["block"].annotation))
    assert OPEN_BLOCKS == {"classification"}
    assert set(closed_blocks()) == every - OPEN_BLOCKS
    assert closed_blocks() == sorted(closed_blocks())


def test_a_custom_model_is_refused(world: dict[str, Any]) -> None:
    config = _classification(world["shared"] / "beans", model={"custom_model": "mine"})
    assert "model.custom_model" in _refused(config, world)


def test_the_run_cannot_write_into_a_shared_folder(world: dict[str, Any]) -> None:
    config = _classification(world["shared"] / "beans")
    moved = config.model_copy(
        update={
            "output": config.output.model_copy(
                update={"models_dir": world["shared"] / "models"}
            )
        }
    )
    with dataset_scope(world["scope"]), pytest.raises(HTTPException) as refusal:
        confine_config(moved, world["profile"])
    assert "output.models_dir" in str(refusal.value.detail)


@pytest.mark.parametrize(
    "name", ["../../secret/weights", "/etc/x", "C:evil", "~/x", "hf_hub:org/model"]
)
def test_a_model_name_that_is_a_path_is_refused(
    world: dict[str, Any], name: str
) -> None:
    config = _classification(world["shared"] / "beans", model={"timm_model": name})
    assert "model.timm_model" in _refused(config, world)


def test_a_plain_model_name_is_accepted(world: dict[str, Any]) -> None:
    config = _classification(
        world["shared"] / "beans", model={"timm_model": "convnext_tiny.fb_in22k"}
    )
    _checked(config, world)


@pytest.fixture
def served(
    world: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[TestClient, list[dict[str, Any]]]]:
    """The real app in server mode; submissions are recorded, never trained."""
    from visionforge.gui.server import app

    submitted: list[dict[str, Any]] = []

    def record(run_id: str, label: str, task: str, *_: Any, **kw: Any) -> Any:
        submitted.append({"run_id": run_id, "task": task, "profile": kw["profile"]})
        from visionforge.gui.api.schemas import RunResponse

        return RunResponse(run_id=run_id, status="running")

    monkeypatch.setattr(routes, "_PROFILES_DIR", tmp_path / "profiles")
    monkeypatch.setattr(routes, "_MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(submission, "_submit_job", record)
    monkeypatch.setattr(
        training_routes, "_check_classification_classes", lambda config: None
    )
    had = hasattr(app.state, "server_mode")
    saved = getattr(app.state, "server_mode", None)
    app.state.server_mode = ServerMode.from_options(
        enabled=True, trusted_proxies=[PROXY], datasets_dirs=[world["shared"]]
    )
    client = TestClient(app, base_url="http://localhost", client=(PROXY, 50000))
    yield client, submitted
    if had:
        app.state.server_mode = saved
    else:
        del app.state.server_mode


def _body(base_dir: Path) -> dict[str, Any]:
    return {
        "name": "beans",
        "model": {"name": "resnet18", "num_classes": 2, "pretrained": False},
        "data": {"base_dir": str(base_dir)},
    }


def test_a_run_on_a_shared_dataset_is_queued_for_who_asked(
    served: tuple[TestClient, list[dict[str, Any]]], world: dict[str, Any]
) -> None:
    client, submitted = served
    response = client.post(
        "/api/experiment/run", json=_body(world["shared"] / "beans"), headers=ANA
    )
    assert response.status_code == 200, response.text
    assert [job["profile"].slug for job in submitted] == [user_slug("ana@lab.org")]


def test_a_run_on_a_folder_outside_the_scope_is_never_queued(
    served: tuple[TestClient, list[dict[str, Any]]], world: dict[str, Any]
) -> None:
    client, submitted = served
    response = client.post(
        "/api/experiment/run", json=_body(world["secret"]), headers=ANA
    )
    assert response.status_code == 422
    assert "data.base_dir" in response.json()["detail"]
    assert submitted == []


def test_a_search_is_never_queued_on_a_server(
    served: tuple[TestClient, list[dict[str, Any]]], world: dict[str, Any]
) -> None:
    client, submitted = served
    body = {
        **_body(world["shared"] / "beans"),
        "block": "grid_search",
        "grid_search": {"hyperparameters": {"data.base_dir": [str(world["secret"])]}},
    }
    response = client.post("/api/experiment/run", json=body, headers=ANA)
    assert response.status_code == 422
    assert "grid_search" in response.json()["detail"]
    assert submitted == []
    assert (
        "grid_search" in client.get("/api/session", headers=ANA).json()["closed_blocks"]
    )
