"""Researcher-defined custom tasks (ADR-058): base contract + registry."""

from visionforge.tasks.base import (
    BaseTaskConfig,
    TaskDataConfig,
    TaskSpec,
    TaskTrainingConfig,
)
from visionforge.tasks.registry import (
    BUILTIN_TASK_KEYS,
    DEFAULT_USER_TASKS_DIR,
    TaskInfo,
    clear_task_registry,
    get_task,
    load_user_tasks,
    register_task,
    registered_tasks,
)

__all__ = [
    "BUILTIN_TASK_KEYS",
    "DEFAULT_USER_TASKS_DIR",
    "BaseTaskConfig",
    "TaskDataConfig",
    "TaskInfo",
    "TaskSpec",
    "TaskTrainingConfig",
    "clear_task_registry",
    "get_task",
    "load_user_tasks",
    "register_task",
    "registered_tasks",
]
