"""What a training config may name on a server (server mode, ADR-215).

The configs of the five tasks live in ``utils/`` and are shared with the
command line, so they cannot carry the server's rule in their field types the
way the dataset requests do (ADR-214). The run routes call
:func:`confine_config` on the validated config instead. With a scope bound it
walks every field and refuses:

- a path outside the person's own profile folder and the shared dataset
  folders; a path the run *writes* must be in the person's own folder;
- a name inside the dataset that climbs out of it (``train_dir: ../x``);
- a model name that is a path: ``YOLO("../x.pt")`` would unpickle that file;
- a custom model, which is code a user brings.

With no scope bound (the local tool) it does nothing.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path, PurePosixPath, PureWindowsPath

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
        "grid_search.hyperparameters",
        "random_search.search_space",
        "transfer_learning.unfreeze_from_layer",
    }
)


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
    own = profile_root(profile)
    for field, value in _leaves(config, ""):
        problem = _problem(field, value, scope, own)
        if problem is not None:
            raise HTTPException(422, problem)


def _leaves(model: BaseModel, prefix: str) -> Iterator[tuple[str, object]]:
    """Every set field of ``model`` and of the models inside it, by dotted name."""
    for name in type(model).model_fields:
        value = getattr(model, name)
        if isinstance(value, BaseModel):
            yield from _leaves(value, f"{prefix}{name}.")
        elif value is not None:
            yield f"{prefix}{name}", value


def _problem(field: str, value: object, scope: DatasetScope, own: Path) -> str | None:
    leaf = field.rsplit(".", 1)[-1]
    if isinstance(value, Path):
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
    written = field.startswith("output.") or field.rsplit(".", 1)[-1].startswith(
        "output"
    )
    if mine or (not written and scope.holds(resolved)):
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
    "DATASET_PARTS",
    "FORBIDDEN",
    "FREE_TEXT",
    "MODEL_NAMES",
    "confine_config",
    "profile_root",
]
