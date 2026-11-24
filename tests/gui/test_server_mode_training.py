"""Server mode: a training config names only what its owner may reach (ADR-215).

The configs are shared with the command line, so the rule is applied to the
validated config by the run routes: paths inside the person's folder or a
shared one, names inside the dataset that only go down, a model name that is
not a path, and no custom model. A strategy rewrites fields per trial, so what
it rewrites is read too (ADR-217).
"""

from __future__ import annotations

import json
import typing
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel

from visionforge.gui.api import routes, strategy_routes, submission, training_routes
from visionforge.gui.api.config_scope import (
    DATASET_PARTS,
    FORBIDDEN,
    FREE_TEXT,
    MODEL_NAMES,
    OPEN_BLOCKS,
    REWRITING,
    closed_blocks,
    confine_axes,
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
from visionforge.utils.selftest_data import build_regression_dataset

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


LOCKED_AXES = [
    "data.base_dir",
    "data.train_dir",
    "model.weights_path",
    "model.custom_model",
    "output.models_dir",
    "classification.checkpoint_path",
    "cross_validation.group_manifest",
    "grid_search.hyperparameters",
    "block",
    "data",
    "model",
    "no.such.field",
]


@pytest.mark.parametrize("axis", LOCKED_AXES)
def test_a_search_axis_cannot_rewrite_a_field_that_names_something(
    world: dict[str, Any], axis: str
) -> None:
    """A grid axis rewrites a field of the config for each trial, after the
    config was confined: it may not name a path, a part of the dataset, a whole
    section or the strategy itself, whatever value it offers."""
    config = _classification(
        world["shared"] / "beans",
        grid_search={"hyperparameters": {axis: [str(world["secret"])]}},
    ).model_copy(update={"block": "grid_search"})
    assert axis in _refused(config, world)
    confine_config(config, world["profile"])  # the local tool keeps its search


def test_a_search_over_numbers_choices_and_model_names_is_accepted(
    world: dict[str, Any],
) -> None:
    config = _classification(
        world["shared"] / "beans",
        grid_search={
            "hyperparameters": {
                "training.learning_rate": [0.001, 0.01],
                "training.batch_size": [8, 16],
                "training.optimizer": ["adam", "sgd"],
                "model.name": ["resnet18", "resnet50"],
                "cross_validation.n_folds": [3, 5],
            }
        },
    ).model_copy(update={"block": "grid_search"})
    _checked(config, world)


def test_a_random_search_cannot_offer_a_model_name_that_is_a_path(
    world: dict[str, Any],
) -> None:
    space = {
        "training.learning_rate": {"type": "log_uniform", "low": 1e-4, "high": 1e-2},
        "model.timm_model": {
            "type": "choice",
            "options": ["convnext_tiny.fb_in22k", "hf_hub:org/model"],
        },
    }
    config = _classification(
        world["shared"] / "beans",
        random_search={"n_trials": 2, "search_space": space},
    ).model_copy(update={"block": "random_search"})
    assert "hf_hub:org/model" in _refused(config, world)


@pytest.mark.parametrize("name", ["../x.pt", "/srv/x.pt", "C:x", "~/x", "s3://b/x"])
def test_an_axis_cannot_offer_a_model_name_that_is_a_path(
    world: dict[str, Any], name: str
) -> None:
    """``YOLO("../x.pt")`` would unpickle that file."""
    with dataset_scope(world["scope"]), pytest.raises(HTTPException) as refusal:
        confine_axes(DetectionConfig, {"model.name": ["yolo11n", name]})
    assert refusal.value.status_code == 422
    assert name in str(refusal.value.detail)
    confine_axes(DetectionConfig, {"model.name": [name]})  # local: nothing bound


@pytest.mark.parametrize("config", CONFIGS)
def test_an_axis_reaches_no_path_and_no_part_of_the_dataset(
    world: dict[str, Any], config: type[BaseModel]
) -> None:
    """Every field of every config is either one an axis may rewrite, and then
    it holds no path, or one it may not."""

    def fields(model: type[BaseModel], prefix: str = "") -> Iterator[tuple[str, Any]]:
        for name, field in model.model_fields.items():
            inner = [
                kind
                for kind in (field.annotation, *typing.get_args(field.annotation))
                if isinstance(kind, type) and issubclass(kind, BaseModel)
            ]
            if inner:
                yield from fields(inner[0], f"{prefix}{name}.")
            else:
                yield f"{prefix}{name}", field.annotation

    open_fields = []
    with dataset_scope(world["scope"]):
        for path, annotation in fields(config):
            try:
                confine_axes(config, {path: ["x"]})
            except HTTPException:
                continue
            open_fields.append(path)
            assert "Path" not in str(annotation), path
            assert path.rsplit(".", 1)[-1] not in DATASET_PARTS, path
            assert not path.endswith("_dir"), path
    assert "training.learning_rate" in open_fields
    assert "model.name" in open_fields


def test_a_group_manifest_inside_the_dataset_is_accepted(world: dict[str, Any]) -> None:
    """The loader resolves a relative manifest against the dataset folder."""
    base = world["shared"] / "beans"
    _checked(
        _classification(base, cross_validation={"group_manifest": "groups.csv"}), world
    )
    climbing = _classification(base, cross_validation={"group_manifest": "../g.csv"})
    assert "g.csv" in _refused(climbing, world)
    outside = _classification(
        base, cross_validation={"group_manifest": str(world["secret"] / "g.csv")}
    )
    assert "cross_validation.group_manifest" in _refused(outside, world)


@pytest.mark.parametrize("block", sorted(closed_blocks()))
def test_the_blocks_that_are_not_training_stay_closed(
    world: dict[str, Any], block: str
) -> None:
    config = _classification(world["shared"] / "beans").model_copy(
        update={"block": block}
    )
    assert block in _refused(config, world)


@pytest.mark.parametrize("block", sorted(OPEN_BLOCKS))
def test_every_training_block_of_classification_is_open(
    world: dict[str, Any], block: str
) -> None:
    config = _classification(world["shared"] / "beans").model_copy(
        update={"block": block}
    )
    _checked(config, world)


def test_the_closed_blocks_are_every_block_but_the_open_ones() -> None:
    every = set(typing.get_args(ExperimentConfig.model_fields["block"].annotation))
    assert set(closed_blocks()) == every - OPEN_BLOCKS
    assert closed_blocks() == ["batch_prediction", "export_onnx"]


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

    async def loaded(*_: Any) -> None:
        return None

    # The runners are built for real; only the wait for torch is skipped.
    monkeypatch.setattr(strategy_routes, "load_training_stack", loaded)
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


def test_a_classification_search_is_queued_when_its_axes_name_nothing(
    served: tuple[TestClient, list[dict[str, Any]]], world: dict[str, Any]
) -> None:
    client, submitted = served
    body = {
        **_body(world["shared"] / "beans"),
        "block": "grid_search",
        "grid_search": {"hyperparameters": {"training.learning_rate": [0.001, 0.01]}},
    }
    response = client.post("/api/experiment/run", json=body, headers=ANA)
    assert response.status_code == 200, response.text
    assert len(submitted) == 1
    assert client.get("/api/session", headers=ANA).json()["closed_blocks"] == [
        "batch_prediction",
        "export_onnx",
    ]


def test_a_classification_search_with_an_axis_out_of_reach_is_never_queued(
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
    assert "data.base_dir" in response.json()["detail"]
    assert submitted == []


def _regression(base: Path) -> dict[str, Any]:
    return {
        "name": "reg",
        "model": {"name": "resnet18", "num_targets": 1, "pretrained": False},
        "data": {"base_dir": str(base), "target_columns": ["target"]},
        "training": {"epochs": 1, "batch_size": 8, "learning_rate": 0.001},
    }


def _strategies(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """One honest request per strategy route of a task."""
    return {
        "compare": {"config": config, "model_names": ["resnet18", "resnet34"]},
        "sweep": {
            "config": config,
            "search_space": {"training.learning_rate": [0.001, 0.01]},
        },
        "cv": {"config": config, "n_folds": 2},
        "replicates": {"config": config, "n_replicates": 2},
        "replicated-comparison": {
            "config": config,
            "variants": {"base": {}, "fast": {"training.learning_rate": 0.01}},
            "n_replicates": 2,
        },
    }


STRATEGIES = ["compare", "sweep", "cv", "replicates", "replicated-comparison"]


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_a_strategy_on_a_shared_dataset_is_queued_for_who_asked(
    served: tuple[TestClient, list[dict[str, Any]]],
    world: dict[str, Any],
    strategy: str,
) -> None:
    client, submitted = served
    base = build_regression_dataset(world["shared"] / "reg")
    body = _strategies(_regression(base))[strategy]
    response = client.post(f"/api/regression/{strategy}", json=body, headers=ANA)
    assert response.status_code == 200, response.text
    assert [job["profile"].slug for job in submitted] == [user_slug("ana@lab.org")]


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_a_strategy_on_a_folder_outside_the_scope_is_never_queued(
    served: tuple[TestClient, list[dict[str, Any]]],
    world: dict[str, Any],
    strategy: str,
) -> None:
    client, submitted = served
    base = build_regression_dataset(world["secret"] / "reg")
    body = _strategies(_regression(base))[strategy]
    response = client.post(f"/api/regression/{strategy}", json=body, headers=ANA)
    assert response.status_code == 422, response.text
    assert "data.base_dir" in response.json()["detail"]
    assert submitted == []


@pytest.mark.parametrize(
    ("strategy", "change"),
    [
        ("compare", {"model_names": ["resnet18", "../../secret/weights"]}),
        ("sweep", {"search_space": {"data.base_dir": ["SECRET"]}}),
        ("sweep", {"search_space": {"model.weights_path": ["SECRET/weights.pt"]}}),
        (
            "sweep",
            {
                "mode": "random",
                "search_space": {
                    "data.images_dir": {"type": "choice", "options": ["../.."]}
                },
            },
        ),
        (
            "replicated-comparison",
            {"variants": {"base": {}, "other": {"data.base_dir": "SECRET"}}},
        ),
        (
            "replicated-comparison",
            {"variants": {"base": {}, "other": {"model.custom_model": "mine"}}},
        ),
    ],
)
def test_a_strategy_cannot_rewrite_its_way_out_of_the_scope(
    served: tuple[TestClient, list[dict[str, Any]]],
    world: dict[str, Any],
    strategy: str,
    change: dict[str, Any],
) -> None:
    """The base config is in reach; what the strategy sets per trial is not."""
    client, submitted = served
    base = build_regression_dataset(world["shared"] / "reg")
    text = json.dumps(change).replace("SECRET", world["secret"].as_posix())
    body = {**_strategies(_regression(base))[strategy], **json.loads(text)}
    response = client.post(f"/api/regression/{strategy}", json=body, headers=ANA)
    assert response.status_code == 422, response.text
    assert submitted == []


def test_the_session_no_longer_lists_the_strategies_as_closed(
    served: tuple[TestClient, list[dict[str, Any]]],
) -> None:
    client, _ = served
    closed = client.get("/api/session", headers=ANA).json()["closed"]
    strategies = [
        key
        for key in closed
        if key.rsplit("/", 1)[-1] in STRATEGIES and "/custom/" not in key
    ]
    assert strategies == []
    assert "POST /api/custom/{key}/sweep" in closed
