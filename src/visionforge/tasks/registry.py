"""Registry + drop-in discovery for researcher-defined tasks (ADR-058).

Mirrors the proven ``user_models/`` pattern (ADR-048): a ``.py`` under
``user_tasks/`` — either ``user_tasks/<name>.py`` or ``user_tasks/<name>/task.py``
— registers a :class:`~visionforge.tasks.base.TaskSpec` subclass with
``@register_task``; ``load_user_tasks()`` imports those files so the decorators
run. The descriptor (key, label, accent color, metric metadata) is everything
the GUI needs to render the task as a real tab — no user-supplied JavaScript.

This executes the researcher's own local Python by design — the trust boundary
is the user's filesystem (ADR-005/048). Nothing is fetched from the network.
"""

from __future__ import annotations

import importlib.util
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from visionforge.tasks.base import TaskSpec

# Keys owned by the built-in task families — a custom task may not shadow them.
BUILTIN_TASK_KEYS = frozenset(
    {"classification", "detection", "regression", "segmentation", "anomaly"}
)

_KEY_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_ACCENT_RE = re.compile(r"^#[0-9a-fA-F]{6}$")

DEFAULT_USER_TASKS_DIR = Path("user_tasks")


@dataclass(frozen=True)
class TaskInfo:
    """Everything the API/GUI needs to surface one registered custom task."""

    key: str
    label: str
    accent: str
    description: str
    metrics: dict[str, str] = field(default_factory=dict)
    primary_metric: str = ""
    spec_cls: type[TaskSpec] | None = None


_REGISTRY: dict[str, TaskInfo] = {}


def register_task(
    *,
    key: str,
    label: str,
    accent: str,
    description: str = "",
    metrics: dict[str, str],
    primary_metric: str,
):
    """Class decorator registering a ``TaskSpec`` subclass as a custom task.

    ``metrics`` maps each metric name the task reports to its direction
    (``"higher"`` or ``"lower"`` is better); ``primary_metric`` must be one of
    them — it drives best-checkpoint selection and report headlines. ``accent``
    is the tab color (``#rrggbb``).

    Raises:
        ValueError: for an invalid key/accent/metrics or a built-in collision.
    """
    if not _KEY_RE.match(key):
        raise ValueError(
            f"Task key '{key}' must be lowercase [a-z0-9_] and start with a letter."
        )
    if key in BUILTIN_TASK_KEYS:
        raise ValueError(f"Task key '{key}' collides with a built-in VisionForge task.")
    if not _ACCENT_RE.match(accent):
        raise ValueError(f"Accent '{accent}' must be a #rrggbb hex color.")
    if not metrics:
        raise ValueError("register_task requires at least one metric.")
    bad = {d for d in metrics.values() if d not in ("higher", "lower")}
    if bad:
        raise ValueError(
            f"Metric directions must be 'higher' or 'lower', got: {sorted(bad)}."
        )
    if primary_metric not in metrics:
        raise ValueError(
            f"primary_metric '{primary_metric}' must be one of {sorted(metrics)}."
        )

    def decorator(spec_cls: type[TaskSpec]) -> type[TaskSpec]:
        existing = _REGISTRY.get(key)
        if existing is not None and existing.spec_cls is not spec_cls:
            logger.warning("Custom task '{}' re-registered; overwriting.", key)
        _REGISTRY[key] = TaskInfo(
            key=key,
            label=label,
            accent=accent,
            description=description,
            metrics=dict(metrics),
            primary_metric=primary_metric,
            spec_cls=spec_cls,
        )
        return spec_cls

    return decorator


def registered_tasks() -> list[TaskInfo]:
    """All currently-registered custom tasks, sorted by key."""
    return [_REGISTRY[k] for k in sorted(_REGISTRY)]


def clear_task_registry() -> None:
    """Drop every registered task. Test hook — not used in the run path."""
    _REGISTRY.clear()


def load_user_tasks(directory: Path | str | None = None) -> list[TaskInfo]:
    """Import task files under ``directory`` so their decorators run.

    Accepts both layouts: ``user_tasks/<name>.py`` and
    ``user_tasks/<name>/task.py``. A missing directory is a no-op, so the
    feature is opt-in; a broken file logs a warning and is skipped.
    """
    directory = Path(directory) if directory is not None else DEFAULT_USER_TASKS_DIR
    if not directory.is_dir():
        return registered_tasks()
    candidates = sorted(directory.glob("*.py")) + sorted(directory.glob("*/task.py"))
    for path in candidates:
        if path.name.startswith("_") or path.parent.name.startswith("_"):
            continue
        _import_file(path)
    return registered_tasks()


def _import_file(path: Path) -> None:
    module_name = f"vf_user_tasks.{path.parent.name}_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        logger.warning("Could not load user-task file: {}", path)
        return
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 — a bad user file must name itself, not crash discovery
        logger.warning("Failed to import user-task file {}: {}", path, exc)


def get_task(key: str, *, user_tasks_dir: Path | str | None = None) -> TaskInfo:
    """Return a registered task, scanning ``user_tasks/`` on first miss.

    Raises:
        KeyError: when ``key`` is not registered even after loading user tasks.
    """
    if key not in _REGISTRY:
        load_user_tasks(user_tasks_dir)
    if key not in _REGISTRY:
        available = [t.key for t in registered_tasks()]
        where = user_tasks_dir if user_tasks_dir is not None else DEFAULT_USER_TASKS_DIR
        raise KeyError(
            f"Custom task '{key}' is not registered "
            f"(available: {available or 'none'}). Define it with @register_task "
            f"in a .py file under {where}/."
        )
    return _REGISTRY[key]


__all__ = [
    "BUILTIN_TASK_KEYS",
    "DEFAULT_USER_TASKS_DIR",
    "TaskInfo",
    "clear_task_registry",
    "get_task",
    "load_user_tasks",
    "register_task",
    "registered_tasks",
]
