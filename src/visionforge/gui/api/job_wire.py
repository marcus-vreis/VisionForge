"""Values that cross between the server and a job's child process as JSON (ADR-252).

Both sides of a strategy's call use it: the server writes the call's arguments
and reads its result; the child reads the arguments and writes the result.
Nothing is pickled. Plain JSON values cross as themselves, a NaN or an
infinity as JSON's ``NaN``/``Infinity`` (which ``json`` reads back); the rest
is a one-key object whose key starts with ``$``:

- ``{"$path": "..."}`` a ``Path``, made absolute where it is written;
- ``{"$tuple": [...]}`` a tuple;
- ``{"$items": [[key, value], ...]}`` a dict whose keys are not all plain
  strings: an int key (a replicated comparison's ``per_seed``), or a string
  that starts with ``$``. So a dict that came in a request is never read as a
  tag: it can only come back as the dict it was.

A dataclass is written as its fields; the reader rebuilds it (``job_call``).
The argument-only tags (a config, a runner, the stop token, the event pump)
are written by ``job_call`` and read by :func:`child_arguments`.

This module imports nothing beyond the standard library: the child loads it
before the training stack.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

# Tags an argument-only hook writes and :func:`child_arguments` reads.
PROGRESS = "$progress"
TOKEN = "$token"
CONFIG = "$config"
MODEL = "$model"
RUNNER = "$runner"

# ``hook(value)`` is the tagged form of an object only it knows, else None.
DumpHook = Callable[[Any], "dict[str, Any] | None"]
# ``read(tag, body)`` is the object an argument-only tag stands for.
TagReader = Callable[[str, Any], Any]


def dump_value(value: Any, hook: DumpHook | None = None) -> Any:
    """``value`` as JSON, with its non-JSON parts tagged.

    Raises:
        TypeError: for a value with no crossing (a callable, an array, an
            unknown object), before anything is sent.
    """
    if hook is not None and (tagged := hook(value)) is not None:
        return tagged
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return {"$path": str(value.expanduser().absolute())}
    if isinstance(value, tuple):
        return {"$tuple": [dump_value(v, hook) for v in value]}
    if isinstance(value, list):
        return [dump_value(v, hook) for v in value]
    if isinstance(value, dict):
        return _dump_dict(value, hook)
    if is_dataclass(value) and not isinstance(value, type):
        return _dump_dict(asdict(value), hook)
    # A numpy scalar (a metric a trainer did not convert): its Python number.
    item = getattr(value, "item", None)
    if callable(item) and getattr(value, "shape", None) == ():
        return dump_value(item(), hook)
    raise TypeError(f"{type(value).__name__} cannot cross to a job's child process")


def _dump_dict(value: dict[Any, Any], hook: DumpHook | None) -> Any:
    if all(isinstance(key, str) and not key.startswith("$") for key in value):
        return {key: dump_value(v, hook) for key, v in value.items()}
    return {
        "$items": [[dump_value(k, hook), dump_value(v, hook)] for k, v in value.items()]
    }


def load_value(value: Any, read: TagReader | None = None) -> Any:
    """The value :func:`dump_value` wrote; ``read`` gives the argument-only tags.

    Raises:
        ValueError: for a tag this side does not read.
    """
    if isinstance(value, list):
        return [load_value(v, read) for v in value]
    if not isinstance(value, dict):
        return value
    tag = next((key for key in value if key.startswith("$")), None)
    if tag is None:
        return {key: load_value(v, read) for key, v in value.items()}
    body = value[tag]
    if tag == "$path":
        return Path(body)
    if tag == "$tuple":
        return tuple(load_value(v, read) for v in body)
    if tag == "$items":
        return {load_value(k, read): load_value(v, read) for k, v in body}
    if read is not None:
        return read(tag, body)
    raise ValueError(f"A job's call carries an unknown tag {tag!r}.")


def child_arguments(
    load: Callable[[str], Any],
    progress: Callable[[dict[str, Any]], None],
    token: Any,
) -> TagReader:
    """How the child reads the argument-only tags: its own pump, token and classes.

    ``load`` imports a ``module:QualName``. A runner is built from its class
    and given the child's token, as ``_run_group_job`` gives it the server's.
    """

    def read(tag: str, body: Any) -> Any:
        if tag == PROGRESS:
            return progress
        if tag == TOKEN:
            return token
        if tag == CONFIG:
            return load_value(body)
        if tag == MODEL:
            return load(body["type"]).model_validate(load_value(body["data"]))
        if tag == RUNNER:
            runner = load(body)()
            if hasattr(runner, "_cancel_token"):
                runner._cancel_token = token
            return runner
        raise ValueError(f"A job's call carries an unknown tag {tag!r}.")

    return read


__all__ = [
    "CONFIG",
    "MODEL",
    "PROGRESS",
    "RUNNER",
    "TOKEN",
    "child_arguments",
    "dump_value",
    "load_value",
]
