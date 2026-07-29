"""Hide or delete a researcher-defined task (ADR-058 follow-up).

Two operations on purpose, because "remove this task" means two very different
things and only one of them is recoverable:

* **hide** — the tab stops rendering; the file stays exactly where it is. This
  is the answer to "my tab bar is full", which is the common case, and it is
  undone by unhiding.
* **delete** — the ``.py`` the researcher wrote is removed from disk. There is
  no undo inside VisionForge, so the API requires the caller to repeat the task
  key as confirmation.

The hidden list lives next to the tasks (``user_tasks/.hidden.json``) rather
than in the per-user config, because the tasks themselves are per working
directory: two projects with a task of the same name must be able to disagree
about whether it is visible.
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from visionforge.tasks.registry import DEFAULT_USER_TASKS_DIR

_HIDDEN_FILE = ".hidden.json"


def _hidden_path(directory: Path | str | None = None) -> Path:
    base = Path(directory) if directory is not None else DEFAULT_USER_TASKS_DIR
    return base / _HIDDEN_FILE


def hidden_tasks(directory: Path | str | None = None) -> set[str]:
    """Keys the researcher has hidden from the task bar."""
    path = _hidden_path(directory)
    if not path.is_file():
        return set()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Could not read {}: {}", path, exc)
        return set()
    if not isinstance(raw, list):
        return set()
    return {str(k) for k in raw}


def _write_hidden(keys: set[str], directory: Path | str | None = None) -> None:
    path = _hidden_path(directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(keys), indent=2), encoding="utf-8")


def set_hidden(key: str, hidden: bool, directory: Path | str | None = None) -> bool:
    """Hide or unhide one task. Returns True when the state actually changed."""
    keys = hidden_tasks(directory)
    if hidden and key not in keys:
        keys.add(key)
    elif not hidden and key in keys:
        keys.discard(key)
    else:
        return False
    _write_hidden(keys, directory)
    logger.info("Task {} {}", key, "hidden" if hidden else "unhidden")
    return True


def task_file(key: str, directory: Path | str | None = None) -> Path | None:
    """The file backing a custom task, in either supported layout."""
    base = Path(directory) if directory is not None else DEFAULT_USER_TASKS_DIR
    flat = base / f"{key}.py"
    if flat.is_file():
        return flat
    packaged = base / key / "task.py"
    if packaged.is_file():
        return packaged
    return None


def delete_task(
    key: str, confirmation: str, directory: Path | str | None = None
) -> Path:
    """Delete a custom task's source file after an explicit confirmation.

    ``confirmation`` must equal ``key``. Typing the name is deliberately more
    work than clicking: this removes code the researcher wrote, and VisionForge
    has no undo for it.

    Returns the path that was removed.

    Raises:
        ValueError: if the confirmation does not match.
        FileNotFoundError: if no file backs that key.
    """
    if confirmation != key:
        raise ValueError(
            f"Confirmation must repeat the task key exactly: expected '{key}'."
        )
    path = task_file(key, directory)
    if path is None:
        raise FileNotFoundError(f"No task file found for '{key}'.")

    # A packaged task owns its folder (it exists to hold assets alongside
    # task.py), so removing only the .py would leave an orphan directory.
    if path.name == "task.py" and path.parent.name == key:
        import shutil

        shutil.rmtree(path.parent)
        removed = path.parent
    else:
        path.unlink()
        removed = path

    # A deleted task should not linger in the hidden list and resurrect the
    # entry if the researcher later re-creates a task with the same key.
    set_hidden(key, False, directory)
    logger.info("Deleted custom task {} at {}", key, removed)
    return removed


__all__ = [
    "delete_task",
    "hidden_tasks",
    "set_hidden",
    "task_file",
]
