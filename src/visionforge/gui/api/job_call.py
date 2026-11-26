"""A strategy's orchestrator called in a server job's child process (ADR-252).

The strategies of ``strategies.py`` (model comparison, sweep, a task's K-fold,
replicates, paired comparison) do not run a Block: each calls one orchestrator
function with a runner or a config, the job's event pump and its stop token.
:func:`call_in_job` makes that call in a worker thread, as before, or, for a
server job (``JobState.isolated``), in the child of ADR-248, with the same
environment, working folder, event pipe, stop line and kill. What the strategy
does with the result (its report, summary files and History group) stays in
the server, after the call returns, in both cases.

The call crosses as JSON (``job_wire``): the function and every class the
child imports are named by :func:`import_path` from objects in the server's
memory (the module attributes ``strategies.py`` looks up, the runner it was
given, the config type it validated), never from a request; a request's values
are data, and a dict of them is escaped so it cannot pass for a tag.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from visionforge.core.cancellation import CancellationToken
from visionforge.gui.api import job_wire
from visionforge.gui.api.job_process import (
    STOP_GRACE_S,
    import_path,
    pinned_config,
    run_in_child,
)
from visionforge.utils.messages import bound_lang

# What a runner may hold and still be rebuilt from its class in the child: the
# job's stop and the orchestrator's event pump, which the child sets itself.
_RUNNER_SLOTS = frozenset({"_cancel_token", "_progress_callback"})


@dataclass(frozen=True)
class RawConfig:
    """A runner's config dict; crossing, it is validated and its paths made absolute.

    The runner's ``config_type`` is read only when the dict crosses: the thread
    gets ``data`` itself, from any runner, as before.
    """

    data: dict[str, Any]
    runner: Any


class ChildCallFailed(RuntimeError):
    """A call that failed in the child; the message is the run's error as worded there."""


def _same(value: Any) -> Any:
    return value


def call_spec(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    progress: Callable[..., Any] | None,
) -> dict[str, Any]:
    """The first line the child reads for ``fn(*args, **kwargs)``.

    Raises:
        TypeError: for a function or an argument that cannot cross.
    """
    if "<" in fn.__qualname__:
        raise TypeError(f"{fn.__qualname__} is not importable by a child process")
    hook = _argument_hook(progress)
    return {
        "kind": "call",
        "function": import_path(fn),
        "args": [job_wire.dump_value(a, hook) for a in args],
        "kwargs": {k: job_wire.dump_value(v, hook) for k, v in kwargs.items()},
        "lang": bound_lang(),
        "grace": STOP_GRACE_S,
    }


def _argument_hook(progress: Callable[..., Any] | None) -> job_wire.DumpHook:
    """Tag the objects only the call carries: pump, token, configs, runner."""

    def hook(value: Any) -> dict[str, Any] | None:
        if progress is not None and value is progress:
            return {job_wire.PROGRESS: True}
        if isinstance(value, CancellationToken):
            return {job_wire.TOKEN: True}
        if isinstance(value, RawConfig):
            model = value.runner.config_type.model_validate(value.data)
            return {job_wire.CONFIG: job_wire.dump_value(pinned_config(model))}
        if isinstance(value, BaseModel):
            data = job_wire.dump_value(pinned_config(value))
            return {job_wire.MODEL: {"type": import_path(type(value)), "data": data}}
        if _is_runner(value):
            return {job_wire.RUNNER: import_path(type(value))}
        return None

    return hook


def _is_runner(value: Any) -> bool:
    """A task runner the child can build from its class.

    Raises:
        TypeError: for a runner that holds state of its own (a custom task's
            holds its task), which its class alone cannot rebuild.
    """
    if isinstance(value, type) or not callable(getattr(value, "run", None)):
        return False
    if not hasattr(value, "config_type"):
        return False
    extra = set(vars(value)) - _RUNNER_SLOTS
    if extra:
        raise TypeError(
            f"{type(value).__name__} holds {sorted(extra)} and cannot be rebuilt "
            "in a child process"
        )
    return True


def _local(value: Any) -> Any:
    return value.data if isinstance(value, RawConfig) else value


async def call_in_job(
    run_id: str,
    fn: Callable[..., Any],
    /,
    *args: Any,
    decode: Callable[[Any], Any] = _same,
    **kwargs: Any,
) -> Any:
    """``fn(*args, **kwargs)`` for job ``run_id``: in a thread, or in a child on a server.

    ``progress_callback`` among ``kwargs`` is the job's event pump; in a child
    it is fed from the child's events, in order. ``decode`` rebuilds what the
    result was before it crossed (a list of trials, a K-fold report); the
    thread's result is returned as it is.

    Raises:
        ChildCallFailed: when the call failed in the child or the child died
            or was killed; the message is the run's error.
        TypeError: when an argument cannot cross.
    """
    from visionforge.gui.api.execution import _job_state

    state = _job_state(run_id)
    if not state.isolated:
        local_args = [_local(a) for a in args]
        local_kwargs = {k: _local(v) for k, v in kwargs.items()}
        return await asyncio.to_thread(fn, *local_args, **local_kwargs)
    progress = kwargs.get("progress_callback")
    spec = call_spec(fn, args, kwargs, progress)
    outcome = await run_in_child(spec, progress or _same, state.token)
    if outcome.status != "completed":
        raise ChildCallFailed(outcome.error)
    return decode(job_wire.load_value(outcome.result))


def trials_of(cls: type[Any]) -> Callable[[Any], list[Any]]:
    """Rebuild a list of ``cls`` trials (comparison, sweep, replicates) from their fields."""

    def decode(rows: Any) -> list[Any]:
        return [cls(**row) for row in rows]

    return decode


def kfold_report(value: Any) -> Any:
    """Rebuild a task K-fold's ``CrossValidationReport`` and its folds."""
    from visionforge.blocks._cv_common import CrossValidationReport, FoldResult

    folds = [FoldResult(**fold) for fold in value["folds"]]
    return CrossValidationReport(**{**value, "folds": folds})


__all__ = [
    "ChildCallFailed",
    "RawConfig",
    "call_in_job",
    "call_spec",
    "kfold_report",
    "trials_of",
]
