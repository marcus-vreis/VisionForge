"""Hiding is reversible, deleting is not — so only one of them is easy."""

from __future__ import annotations

from pathlib import Path

import pytest

from visionforge.tasks.manage import (
    delete_task,
    hidden_tasks,
    set_hidden,
    task_file,
)


@pytest.fixture
def tasks_dir(tmp_path: Path) -> Path:
    d = tmp_path / "user_tasks"
    d.mkdir()
    (d / "flat_task.py").write_text("# a task", encoding="utf-8")
    packaged = d / "packaged_task"
    packaged.mkdir()
    (packaged / "task.py").write_text("# a task", encoding="utf-8")
    (packaged / "asset.txt").write_text("data the task needs", encoding="utf-8")
    return d


class TestHiding:
    def test_hidden_starts_empty(self, tasks_dir: Path) -> None:
        assert hidden_tasks(tasks_dir) == set()

    def test_hide_then_read_back(self, tasks_dir: Path) -> None:
        assert set_hidden("flat_task", True, tasks_dir) is True
        assert hidden_tasks(tasks_dir) == {"flat_task"}

    def test_hiding_leaves_the_file_alone(self, tasks_dir: Path) -> None:
        """The whole point of hiding: reclaim a tab without risking the code."""
        set_hidden("flat_task", True, tasks_dir)
        assert task_file("flat_task", tasks_dir) is not None

    def test_unhide(self, tasks_dir: Path) -> None:
        set_hidden("flat_task", True, tasks_dir)
        assert set_hidden("flat_task", False, tasks_dir) is True
        assert hidden_tasks(tasks_dir) == set()

    def test_repeating_the_same_state_reports_no_change(self, tasks_dir: Path) -> None:
        set_hidden("flat_task", True, tasks_dir)
        assert set_hidden("flat_task", True, tasks_dir) is False

    def test_hiding_one_does_not_hide_another(self, tasks_dir: Path) -> None:
        set_hidden("flat_task", True, tasks_dir)
        assert "packaged_task" not in hidden_tasks(tasks_dir)

    def test_a_corrupt_hidden_file_reads_as_nothing_hidden(
        self, tasks_dir: Path
    ) -> None:
        (tasks_dir / ".hidden.json").write_text("[[[", encoding="utf-8")
        assert hidden_tasks(tasks_dir) == set()


class TestTaskFile:
    def test_finds_the_flat_layout(self, tasks_dir: Path) -> None:
        assert task_file("flat_task", tasks_dir) == tasks_dir / "flat_task.py"

    def test_finds_the_packaged_layout(self, tasks_dir: Path) -> None:
        found = task_file("packaged_task", tasks_dir)
        assert found == tasks_dir / "packaged_task" / "task.py"

    def test_unknown_key_is_none(self, tasks_dir: Path) -> None:
        assert task_file("nope", tasks_dir) is None


class TestDeleting:
    def test_requires_the_key_as_confirmation(self, tasks_dir: Path) -> None:
        with pytest.raises(ValueError, match="repeat the task key"):
            delete_task("flat_task", "yes", tasks_dir)
        assert task_file("flat_task", tasks_dir) is not None

    def test_a_near_miss_is_still_refused(self, tasks_dir: Path) -> None:
        with pytest.raises(ValueError):
            delete_task("flat_task", "flat_tas", tasks_dir)
        assert task_file("flat_task", tasks_dir) is not None

    def test_correct_confirmation_removes_the_flat_file(self, tasks_dir: Path) -> None:
        removed = delete_task("flat_task", "flat_task", tasks_dir)
        assert removed == tasks_dir / "flat_task.py"
        assert task_file("flat_task", tasks_dir) is None

    def test_packaged_task_takes_its_folder_with_it(self, tasks_dir: Path) -> None:
        """A packaged task owns its directory — it exists to hold assets next
        to task.py, so removing only the .py would orphan them."""
        delete_task("packaged_task", "packaged_task", tasks_dir)
        assert not (tasks_dir / "packaged_task").exists()

    def test_missing_task_raises(self, tasks_dir: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No task file"):
            delete_task("nope", "nope", tasks_dir)

    def test_deleting_clears_a_stale_hidden_entry(self, tasks_dir: Path) -> None:
        """Otherwise re-creating a task with the same key would resurrect it
        already hidden, with nothing on screen to explain why."""
        set_hidden("flat_task", True, tasks_dir)
        delete_task("flat_task", "flat_task", tasks_dir)
        assert hidden_tasks(tasks_dir) == set()
