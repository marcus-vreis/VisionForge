from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import pytest
from pydantic import Field

from visionforge.tasks import (
    BaseTaskConfig,
    TaskSpec,
    clear_task_registry,
    get_task,
    load_user_tasks,
    register_task,
    registered_tasks,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    clear_task_registry()
    yield
    clear_task_registry()


def _toy_spec(key: str = "counting") -> type[TaskSpec]:
    @register_task(
        key=key,
        label="Contagem",
        accent="#2dd4bf",
        description="toy",
        metrics={"mae": "lower", "rmse": "lower"},
        primary_metric="mae",
    )
    class ToySpec(TaskSpec):
        def build_model(self, cfg: Any) -> Any:
            return object()

        def build_loaders(self, cfg: Any) -> tuple[Any, Any, None]:
            return [], [], None

        def compute_loss(self, model: Any, batch: Any, cfg: Any) -> Any:
            return 0.0

        def compute_metrics(self, model: Any, loader: Any, cfg: Any) -> dict:
            return {"mae": 0.0, "rmse": 0.0}

    return ToySpec


class TestRegisterTask:
    def test_roundtrip_via_get_task(self) -> None:
        spec_cls = _toy_spec()
        info = get_task("counting")
        assert info.label == "Contagem"
        assert info.accent == "#2dd4bf"
        assert info.primary_metric == "mae"
        assert info.metrics == {"mae": "lower", "rmse": "lower"}
        assert info.spec_cls is spec_cls
        assert [t.key for t in registered_tasks()] == ["counting"]

    def test_builtin_key_collision_rejected(self) -> None:
        with pytest.raises(ValueError, match="built-in"):
            _toy_spec(key="segmentation")

    @pytest.mark.parametrize("key", ["Bad", "1abc", "has-dash", ""])
    def test_invalid_keys_rejected(self, key: str) -> None:
        with pytest.raises(ValueError, match="key"):
            _toy_spec(key=key)

    def test_invalid_accent_rejected(self) -> None:
        with pytest.raises(ValueError, match="hex"):
            register_task(
                key="x1",
                label="X",
                accent="teal",
                metrics={"m": "higher"},
                primary_metric="m",
            )

    def test_primary_metric_must_be_declared(self) -> None:
        with pytest.raises(ValueError, match="primary_metric"):
            register_task(
                key="x2",
                label="X",
                accent="#112233",
                metrics={"m": "higher"},
                primary_metric="other",
            )

    def test_metric_direction_validated(self) -> None:
        with pytest.raises(ValueError, match="direction"):
            register_task(
                key="x3",
                label="X",
                accent="#112233",
                metrics={"m": "best"},
                primary_metric="m",
            )

    def test_unknown_task_raises_with_guidance(self) -> None:
        with pytest.raises(KeyError, match="register_task"):
            get_task("nope", user_tasks_dir="does_not_exist")


class TestUserTasksDiscovery:
    def test_loads_flat_file_and_package_layouts(self, tmp_path: Path) -> None:
        flat = tmp_path / "flat_task.py"
        flat.write_text(
            textwrap.dedent(
                """
                from typing import Any
                from visionforge.tasks import TaskSpec, register_task

                @register_task(
                    key="flat",
                    label="Flat",
                    accent="#aabbcc",
                    metrics={"score": "higher"},
                    primary_metric="score",
                )
                class Flat(TaskSpec):
                    def build_model(self, cfg: Any) -> Any: return object()
                    def build_loaders(self, cfg: Any): return [], [], None
                    def compute_loss(self, model, batch, cfg): return 0.0
                    def compute_metrics(self, model, loader, cfg): return {"score": 1.0}
                """
            ),
            encoding="utf-8",
        )
        pkg = tmp_path / "nested" / "task.py"
        pkg.parent.mkdir()
        pkg.write_text(
            flat.read_text(encoding="utf-8")
            .replace('key="flat"', 'key="nested"')
            .replace('label="Flat"', 'label="Nested"'),
            encoding="utf-8",
        )

        infos = load_user_tasks(tmp_path)
        assert [t.key for t in infos] == ["flat", "nested"]

    def test_broken_file_is_skipped_not_fatal(self, tmp_path: Path) -> None:
        (tmp_path / "broken.py").write_text("raise RuntimeError('boom')\n", "utf-8")
        assert load_user_tasks(tmp_path) == []

    def test_missing_directory_is_noop(self, tmp_path: Path) -> None:
        assert load_user_tasks(tmp_path / "absent") == []


class TestBaseTaskConfig:
    def test_defaults_validate_and_carry_schema_version(self, tmp_path: Path) -> None:
        cfg = BaseTaskConfig.model_validate({"data": {"base_dir": str(tmp_path)}})
        assert cfg.schema_version == 1
        assert cfg.training.epochs == 10
        # True since ADR-098: a custom task's runs reproduce like every other.
        assert cfg.training.deterministic is True
        assert cfg.data.transforms.horizontal_flip is True

    def test_subclass_extends_with_task_fields(self, tmp_path: Path) -> None:
        class CountingConfig(BaseTaskConfig):
            density_sigma: float = Field(default=2.0, gt=0)

        cfg = CountingConfig.model_validate(
            {"data": {"base_dir": str(tmp_path)}, "density_sigma": 3.5}
        )
        assert cfg.density_sigma == 3.5
        assert "density_sigma" in CountingConfig.model_json_schema()["properties"]

    def test_has_custom_run_detects_level2(self) -> None:
        spec_cls = _toy_spec("lvl1")
        assert spec_cls().has_custom_run() is False

        @register_task(
            key="lvl2",
            label="L2",
            accent="#334455",
            metrics={"m": "higher"},
            primary_metric="m",
        )
        class Level2(TaskSpec):
            def build_model(self, cfg: Any) -> Any:
                return object()

            def build_loaders(self, cfg: Any):
                return [], [], None

            def compute_loss(self, model: Any, batch: Any, cfg: Any) -> Any:
                return 0.0

            def compute_metrics(self, model: Any, loader: Any, cfg: Any) -> dict:
                return {"m": 1.0}

            def run(self, cfg: Any, ctx: Any) -> dict:
                return {"m": 1.0}

        assert Level2().has_custom_run() is True
