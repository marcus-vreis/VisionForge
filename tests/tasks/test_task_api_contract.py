"""The extension points a researcher's ``user_tasks/`` and ``user_models/`` code uses.

Renaming a parameter or dropping a hook breaks every custom task in the wild
without a single built-in test noticing (ADR-190). These tests pin the names,
kinds and defaults the docs under ``docs/custom/`` promise; changing one is a
deliberate API change, made together with the Stability section of those guides.
"""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Callable
from typing import Any

from visionforge import tasks
from visionforge.models import registry as model_registry
from visionforge.tasks import engine
from visionforge.tasks.base import BaseTaskConfig, TaskSpec

Shape = list[tuple[str, str, str]]

POSITIONAL = "POSITIONAL_OR_KEYWORD"
KEYWORD = "KEYWORD_ONLY"
REQUIRED = "-"


def shape(func: Callable[..., Any]) -> Shape:
    """(name, kind, default) per parameter; annotations are not part of the contract."""
    return [
        (
            p.name,
            p.kind.name,
            REQUIRED if p.default is inspect.Parameter.empty else repr(p.default),
        )
        for p in inspect.signature(func).parameters.values()
    ]


def required(*names: str) -> Shape:
    """Required positional-or-keyword parameters, in order."""
    return [(name, POSITIONAL, REQUIRED) for name in names]


class _Probe(TaskSpec[BaseTaskConfig]):
    """The smallest concrete spec: the four hooks, no Level 2 ``run``."""

    def build_model(self, cfg: BaseTaskConfig) -> Any:
        raise NotImplementedError

    def build_loaders(self, cfg: BaseTaskConfig) -> Any:
        raise NotImplementedError

    def compute_loss(self, model: Any, batch: Any, cfg: BaseTaskConfig) -> Any:
        raise NotImplementedError

    def compute_metrics(self, model: Any, loader: Any, cfg: BaseTaskConfig) -> Any:
        raise NotImplementedError


def test_taskspec_has_exactly_the_four_documented_abstract_hooks() -> None:
    assert TaskSpec.__abstractmethods__ == {
        "build_model",
        "build_loaders",
        "compute_loss",
        "compute_metrics",
    }


def test_taskspec_hook_signatures() -> None:
    assert shape(TaskSpec.build_model) == required("self", "cfg")
    assert shape(TaskSpec.build_loaders) == required("self", "cfg")
    assert shape(TaskSpec.compute_loss) == required("self", "model", "batch", "cfg")
    assert shape(TaskSpec.compute_metrics) == required("self", "model", "loader", "cfg")


def test_level_2_run_hook_is_optional_and_defaults_to_the_engine() -> None:
    assert shape(TaskSpec.run) == required("self", "cfg", "ctx")
    assert "run" not in TaskSpec.__abstractmethods__
    spec = _Probe()
    assert spec.run(None, None) is None  # type: ignore[arg-type]
    assert not spec.has_custom_run()


def test_taskspec_is_configured_through_a_config_class_var() -> None:
    assert TaskSpec.Config is BaseTaskConfig


def test_base_task_config_fields() -> None:
    assert list(BaseTaskConfig.model_fields) == [
        "schema_version",
        "name",
        "data",
        "training",
        "output",
        "device",
    ]


def test_register_task_arguments_are_keyword_only() -> None:
    assert shape(tasks.register_task) == [
        ("key", KEYWORD, REQUIRED),
        ("label", KEYWORD, REQUIRED),
        ("accent", KEYWORD, REQUIRED),
        ("description", KEYWORD, "''"),
        ("metrics", KEYWORD, REQUIRED),
        ("primary_metric", KEYWORD, REQUIRED),
    ]


def test_task_info_fields() -> None:
    assert [f.name for f in dataclasses.fields(tasks.TaskInfo)] == [
        "key",
        "label",
        "accent",
        "description",
        "metrics",
        "primary_metric",
        "spec_cls",
    ]


def test_level_2_context_exposes_what_the_guide_lists() -> None:
    assert [f.name for f in dataclasses.fields(engine.TaskRunContext)] == [
        "run_dir",
        "device",
        "device_label",
        "emit",
    ]
    assert shape(engine.TaskRunContext.save_checkpoint) == [
        *required("self", "model"),
        ("name", POSITIONAL, "'best_model.pth'"),
    ]


def test_package_exports_the_names_the_guide_imports() -> None:
    assert {
        "BaseTaskConfig",
        "TaskDataConfig",
        "TaskSpec",
        "TaskTrainingConfig",
        "register_task",
    } <= set(tasks.__all__)


def test_user_model_registration_contract() -> None:
    assert shape(model_registry.register_model) == required("name")
    assert shape(model_registry.build_custom_model) == [
        *required("name"),
        ("num_outputs", KEYWORD, REQUIRED),
        ("user_models_dir", KEYWORD, "None"),
    ]
    assert {"register_model", "build_custom_model", "registered_models"} <= set(
        model_registry.__all__
    )
