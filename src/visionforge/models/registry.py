"""User-supplied custom model registry — drop-in ``user_models/`` (ADR-048/049).

A researcher registers a custom architecture with ``@register_model("name")`` in a
``.py`` file under a user-models directory (default ``./user_models``).
``load_user_models()`` imports those files so their decorators run, after which
``build_custom_model("name", num_outputs=...)`` returns a fresh ``nn.Module``.

The builder receives a single int — the task's output dimension (``num_classes``
for classification/segmentation, ``num_targets`` for regression) — so one custom
model can serve every CNN-headed task.

This executes the researcher's own local Python by design — VisionForge is
local-first and runs on the user's own machine (ADR-005), so the trust boundary is
the user's filesystem. Nothing is fetched from the network.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from pathlib import Path

import torch.nn as nn
from loguru import logger

# A builder receives the task's output dimension and returns a ready model.
ModelBuilder = Callable[[int], nn.Module]

_REGISTRY: dict[str, ModelBuilder] = {}

DEFAULT_USER_MODELS_DIR = Path("user_models")


def register_model(name: str) -> Callable[[ModelBuilder], ModelBuilder]:
    """Decorator registering a custom model builder under ``name``.

    The builder is called as ``builder(num_outputs)`` — the task's output
    dimension — and must return an ``nn.Module`` emitting that many outputs.
    """
    if not name or not name.strip():
        raise ValueError("register_model requires a non-empty name.")

    def decorator(builder: ModelBuilder) -> ModelBuilder:
        if name in _REGISTRY and _REGISTRY[name] is not builder:
            logger.warning("Custom model '{}' re-registered; overwriting.", name)
        _REGISTRY[name] = builder
        return builder

    return decorator


def registered_models() -> list[str]:
    """Names of all currently-registered custom models (sorted)."""
    return sorted(_REGISTRY)


def clear_registry() -> None:
    """Drop every registered builder. Test hook — not used in the run path."""
    _REGISTRY.clear()


def load_user_models(directory: Path | str | None = None) -> list[str]:
    """Import every non-underscore ``.py`` file in ``directory`` to run its
    ``@register_model`` decorators, and return the registered names.

    A missing directory is a no-op (returns whatever is already registered), so
    the feature is opt-in: with no ``user_models/`` folder nothing changes.
    """
    directory = Path(directory) if directory is not None else DEFAULT_USER_MODELS_DIR
    if not directory.is_dir():
        return registered_models()
    for path in sorted(directory.glob("*.py")):
        if path.name.startswith("_"):
            continue
        _import_file(path)
    return registered_models()


def _import_file(path: Path) -> None:
    spec = importlib.util.spec_from_file_location(f"vf_user_models.{path.stem}", path)
    if spec is None or spec.loader is None:
        logger.warning("Could not load user-model file: {}", path)
        return
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 — a bad user file must name itself, not crash discovery
        logger.warning("Failed to import user-model file {}: {}", path, exc)


def build_custom_model(
    name: str,
    *,
    num_outputs: int,
    user_models_dir: Path | str | None = None,
) -> nn.Module:
    """Build a registered custom model, scanning ``user_models/`` on first miss.

    ``num_outputs`` is the task's output dimension passed to the builder.

    Raises:
        KeyError: when ``name`` is not registered even after loading user models.
    """
    if name not in _REGISTRY:
        load_user_models(user_models_dir)
    if name not in _REGISTRY:
        available = registered_models()
        where = (
            user_models_dir if user_models_dir is not None else DEFAULT_USER_MODELS_DIR
        )
        raise KeyError(
            f"Custom model '{name}' is not registered "
            f"(available: {available or 'none'}). Define it with @register_model "
            f"in a .py file under {where}/."
        )
    return _REGISTRY[name](num_outputs)


__all__ = [
    "ModelBuilder",
    "DEFAULT_USER_MODELS_DIR",
    "register_model",
    "registered_models",
    "clear_registry",
    "load_user_models",
    "build_custom_model",
]
