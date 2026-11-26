"""What a training config may name on a server (server mode, ADR-215, ADR-217).

The configs of the five tasks live in ``utils/`` and are shared with the
command line, so they cannot carry the server's rule in their field types the
way the dataset requests do (ADR-214). The run routes call
:func:`confine_config` on the validated config instead. With a scope bound it
walks every field and refuses:

- a path outside the person's own profile folder and the shared dataset
  folders; a path the run *writes* must be in the person's own folder;
- a name inside the dataset that climbs out of it (``train_dir: ../x``);
- a model name that is a path: ``YOLO("../x.pt")`` would unpickle that file;
- a custom model, which is code a user brings;
- for classification, the two ``block`` values that are not training
  (``batch_prediction``, ``export_onnx``): they are read with the evaluation.

A search, a comparison and a set of variants rewrite fields of the config for
each trial, after that walk. :func:`confine_axes` reads what they rewrite: an
axis may only name a field that holds numbers, switches or choices, and a model
name it offers is checked like the field itself. Paths, names inside the
dataset, ``block`` and the search spaces themselves cannot be rewritten.

With no scope bound (the local tool) neither does anything.

Every config a server runs passes :func:`confine_config`, so it also notes the
folders the config reads, for :func:`take_reads`: the job that request queues
learns which of the person's own datasets it uses (ADR-249).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextvars import ContextVar
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import UnionType
from typing import Annotated, Literal, Union, get_args, get_origin

from fastapi import HTTPException
from pydantic import BaseModel

from visionforge.gui.api.dataset_scope import DatasetScope, current_scope
from visionforge.gui.api.profiles import Profile
from visionforge.utils.messages import tr

# Text fields that name something inside the dataset folder.
DATASET_PARTS = frozenset(
    {
        "train_dir",
        "val_dir",
        "test_dir",
        "images_dir",
        "images_subdir",
        "masks_subdir",
        "normal_dir",
        "train_csv",
        "val_csv",
        "test_csv",
    }
)

# Text fields a library resolves as an architecture name, and as a file when
# they look like one.
MODEL_NAMES = frozenset({"model.name", "model.timm_model"})

# Text fields that have no place on a server.
FORBIDDEN = frozenset({"model.custom_model"})

# Text fields read and found to name nothing on the disk. A new text field of a
# config must be added to one of these four sets, or the suite fails.
FREE_TEXT = frozenset(
    {
        "name",
        "data.class_names",
        "data.image_column",
        "data.target_columns",
        "batch_prediction.class_names",
        "batch_prediction.image_extensions",
        "transfer_learning.unfreeze_from_layer",
    }
)


# Fields whose keys are dotted names of other fields, set for each trial
# (``{"data.base_dir": [...]}``): what they name is whatever those fields name,
# so their keys are read by ``confine_axes``.
REWRITING = frozenset({"grid_search.hyperparameters", "random_search.search_space"})

# Path fields the loader resolves against the dataset folder and refuses
# outside it (``load_group_manifest``): a relative one only has to go down.
DATASET_FILES = frozenset({"cross_validation.group_manifest"})

# Fields no axis may rewrite, whatever they hold: the strategy itself.
_LOCKED = FORBIDDEN | REWRITING | {"block"}

# The values of classification's ``block`` a server runs. All go through one
# route, ``/api/experiment/run``, so the gate cannot tell them apart. The two
# left out are not training: they are read with the evaluation routes.
OPEN_BLOCKS = frozenset(
    {
        "classification",
        "cross_validation",
        "grid_search",
        "model_comparison",
        "random_search",
        "transfer_learning",
    }
)

# What a field an axis may rewrite is made of. Anything else (a path, a dict,
# a model, ``Any``) is refused: the rule fails closed for a field added later.
_VALUES = (int, float, bool, str, type(None))
_CONTAINERS = (list, tuple, Union, UnionType)


# The folders the configs confined while serving this request read (ADR-249).
_READS: ContextVar[tuple[Path, ...]] = ContextVar(
    "visionforge_config_reads", default=()
)


def take_reads() -> tuple[Path, ...]:
    """The resolved paths the configs confined in this request read; then none.

    The request that queues a job takes them, so the queue knows which of the
    person's own datasets the job uses. Empty on a local install.
    """
    reads = _READS.get()
    _READS.set(())
    return reads


def closed_blocks() -> list[str]:
    """The values of classification's ``block`` a server refuses, for the screen."""
    from visionforge.utils.config import ExperimentConfig

    every = get_args(ExperimentConfig.model_fields["block"].annotation)
    return sorted(set(every) - OPEN_BLOCKS)


def profile_root(profile: Profile) -> Path:
    """The folder that holds everything of ``profile``: outputs and datasets."""
    root = profile.root if profile.root is not None else profile.models_dir.parent
    return root.resolve()


def confine_config(config: BaseModel, profile: Profile) -> None:
    """Refuse ``config`` when it names something outside the person's reach.

    Raises:
        HTTPException: 422 naming the field, in server mode only.
    """
    scope = current_scope()
    if scope is None:
        return
    block = getattr(config, "block", None)
    if isinstance(block, str) and block not in OPEN_BLOCKS:
        raise HTTPException(422, tr("server.config_block", block=block))
    own = profile_root(profile)
    reads: list[Path] = []
    for field, value in _leaves(config, ""):
        problem = _problem(field, value, scope, own)
        if problem is not None:
            raise HTTPException(422, problem)
        if field in REWRITING and isinstance(value, Mapping):
            confine_axes(type(config), value)
        read = _read_path(field, value)
        if read is not None:
            reads.append(read)
    _READS.set((*_READS.get(), *reads))


def confine_axes(config_type: type[BaseModel], axes: Mapping[str, object]) -> None:
    """Refuse ``axes`` when one rewrites a field that names something.

    ``axes`` maps a dotted field name to what a strategy sets it to: a grid's
    list of values, a random search's spec, one variant's value.

    Raises:
        HTTPException: 422 naming the axis, in server mode only.
    """
    if current_scope() is None:
        return
    for path, offered in axes.items():
        problem = _axis_problem(config_type, str(path), offered)
        if problem is not None:
            raise HTTPException(422, problem)


def _axis_problem(
    config_type: type[BaseModel], path: str, offered: object
) -> str | None:
    kind = _annotation_at(config_type, path)
    leaf = path.rsplit(".", 1)[-1]
    if path in _LOCKED or leaf in DATASET_PARTS or not _holds_values(kind):
        return tr("server.config_axis", field=path[:80])
    if path in MODEL_NAMES:
        for text in _texts(offered):
            if not _plain_name(text):
                return tr("server.config_name", field=path, value=text[:80])
    return None


def _annotation_at(config_type: type[BaseModel], path: str) -> object:
    """The annotation of the field ``path`` names, or ``None`` when it names none."""
    kind: object = config_type
    for part in path.split("."):
        model = _model_in(kind)
        if model is None or part not in model.model_fields:
            return None
        kind = model.model_fields[part].annotation
    return kind


def _model_in(kind: object) -> type[BaseModel] | None:
    """``kind`` when it is a config model, or the one inside ``Model | None``."""
    if isinstance(kind, type) and issubclass(kind, BaseModel):
        return kind
    if get_origin(kind) in (Union, UnionType):
        models = [m for m in map(_model_in, get_args(kind)) if m is not None]
        return models[0] if len(models) == 1 else None
    return None


def _holds_values(kind: object) -> bool:
    """True when a field annotated ``kind`` holds only numbers, text and choices."""
    origin = get_origin(kind)
    if origin is Literal:
        return True
    if origin is Annotated:
        return _holds_values(get_args(kind)[0])
    if origin in _CONTAINERS:
        return all(_holds_values(a) for a in get_args(kind) if a is not Ellipsis)
    return kind in _VALUES or (isinstance(kind, type) and issubclass(kind, Enum))


def _texts(offered: object) -> Iterator[str]:
    """Every text inside what an axis offers: a value, a list, a spec's options."""
    if isinstance(offered, str):
        yield offered
    elif isinstance(offered, Mapping):
        for value in offered.values():
            yield from _texts(value)
    elif isinstance(offered, list | tuple):
        for value in offered:
            yield from _texts(value)


def _leaves(model: BaseModel, prefix: str) -> Iterator[tuple[str, object]]:
    """Every set field of ``model`` and of the models inside it, by dotted name."""
    for name in type(model).model_fields:
        value = getattr(model, name)
        if isinstance(value, BaseModel):
            yield from _leaves(value, f"{prefix}{name}.")
        elif value is not None:
            yield f"{prefix}{name}", value


def _written(field: str) -> bool:
    """True for a path the run writes: ``output.*`` and ``*.output*``."""
    return field.startswith("output.") or field.rsplit(".", 1)[-1].startswith("output")


def _read_path(field: str, value: object) -> Path | None:
    """The resolved folder or file a path field makes the run read, if any.

    A relative file the loader resolves inside the dataset folder adds nothing:
    the dataset folder is already read.
    """
    if not isinstance(value, Path) or _written(field):
        return None
    if field in DATASET_FILES and not value.is_absolute():
        return None
    try:
        return value.expanduser().resolve()
    except (OSError, RuntimeError):
        return None


def _problem(field: str, value: object, scope: DatasetScope, own: Path) -> str | None:
    leaf = field.rsplit(".", 1)[-1]
    if isinstance(value, Path):
        if field in DATASET_FILES and not value.is_absolute():
            down = _only_goes_down(str(value))
            return None if down else tr("server.dataset_part", value=str(value)[:80])
        return _path_problem(field, value, scope, own)
    if not isinstance(value, str):
        return None
    if field in FORBIDDEN:
        return tr("server.config_forbidden", field=field)
    if leaf in DATASET_PARTS and not _only_goes_down(value):
        return tr("server.dataset_part", value=value[:80])
    if field in MODEL_NAMES and not _plain_name(value):
        return tr("server.config_name", field=field, value=value[:80])
    return None


def _path_problem(
    field: str, value: Path, scope: DatasetScope, own: Path
) -> str | None:
    try:
        resolved = value.expanduser().resolve()
    except (OSError, RuntimeError):
        return tr("server.config_path", field=field)
    mine = resolved == own or resolved.is_relative_to(own)
    if mine or (not _written(field) and scope.holds(resolved)):
        return None
    return tr("server.config_path", field=field)


def _only_goes_down(text: str) -> bool:
    windows, posix = PureWindowsPath(text), PurePosixPath(text)
    climbs = ".." in windows.parts or ".." in posix.parts
    rooted = bool(windows.drive or windows.root or posix.root)
    return not (climbs or rooted or text.strip().startswith("~"))


def _plain_name(text: str) -> bool:
    """True for ``yolo11n`` or ``convnext_tiny.fb_in22k``; false for anything a
    library could read as a path, a drive or a remote address."""
    return not (
        set(text) & {"/", "\\", ":"} or text.startswith((".", "~")) or ".." in text
    )


__all__ = [
    "DATASET_FILES",
    "DATASET_PARTS",
    "FORBIDDEN",
    "FREE_TEXT",
    "MODEL_NAMES",
    "OPEN_BLOCKS",
    "REWRITING",
    "closed_blocks",
    "confine_axes",
    "confine_config",
    "profile_root",
    "take_reads",
]
