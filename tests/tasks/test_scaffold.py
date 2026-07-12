"""The scaffolded template and the shipped example must be FUNCTIONAL:
these tests import them through the real discovery path and train them
through the real engine, so a template that stops working fails CI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from visionforge.tasks import clear_task_registry, get_task, load_user_tasks
from visionforge.tasks.engine import GenericTaskEngine
from visionforge.tasks.scaffold import scaffold_task

REPO_USER_TASKS = Path(__file__).parents[2] / "user_tasks"


@pytest.fixture(autouse=True)
def _clean_registry():
    clear_task_registry()
    yield
    clear_task_registry()


def _config_dict(tmp_path: Path, **extra: Any) -> dict[str, Any]:
    return {
        "name": "scaffold_run",
        "data": {"base_dir": str(tmp_path)},
        "training": {"epochs": 1, "batch_size": 16},
        "output": {"models_dir": str(tmp_path / "models")},
        "device": {"kind": "cpu"},
        **extra,
    }


class TestScaffoldTask:
    def test_creates_flat_file(self, tmp_path: Path) -> None:
        target = scaffold_task("my_task", directory=tmp_path)
        assert target == tmp_path / "my_task.py"
        content = target.read_text(encoding="utf-8")
        assert 'key="my_task"' in content
        assert "class MyTaskConfig" in content

    def test_package_layout(self, tmp_path: Path) -> None:
        target = scaffold_task("my_task", directory=tmp_path, package=True)
        assert target == tmp_path / "my_task" / "task.py"
        assert target.is_file()

    def test_invalid_key_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="lowercase"):
            scaffold_task("My-Task", directory=tmp_path)

    def test_builtin_key_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="built-in"):
            scaffold_task("classification", directory=tmp_path)

    def test_existing_file_requires_force(self, tmp_path: Path) -> None:
        scaffold_task("my_task", directory=tmp_path)
        with pytest.raises(FileExistsError):
            scaffold_task("my_task", directory=tmp_path)
        # force=True overwrites without complaint
        scaffold_task("my_task", directory=tmp_path, force=True)

    def test_scaffolded_template_registers_and_trains(self, tmp_path: Path) -> None:
        """The generated file must work untouched: discovery + a real run."""
        scaffold_task("fresh_task", directory=tmp_path / "user_tasks")
        load_user_tasks(tmp_path / "user_tasks")

        info = get_task("fresh_task", user_tasks_dir=tmp_path / "user_tasks")
        assert info.label == "Fresh task"
        assert info.primary_metric == "mae"
        assert info.spec_cls is not None

        cfg = info.spec_cls.Config.model_validate(_config_dict(tmp_path, n_samples=32))
        result = GenericTaskEngine(info, cfg).run()
        assert "mae" in result.metrics
        assert (result.run_dir / "run.json").is_file()


class TestExampleCounting:
    def test_ships_and_trains(self, tmp_path: Path) -> None:
        load_user_tasks(REPO_USER_TASKS)
        info = get_task("example_counting", user_tasks_dir=REPO_USER_TASKS)
        assert info.accent == "#2dd4bf"
        assert info.metrics == {"mae": "lower", "rmse": "lower"}
        assert info.spec_cls is not None

        cfg = info.spec_cls.Config.model_validate(
            _config_dict(tmp_path, n_samples=32, max_count=3)
        )
        result = GenericTaskEngine(info, cfg).run()
        assert {"mae", "rmse", "test_mae", "test_rmse"} <= set(result.metrics)
        assert result.metrics["mae"] >= 0
