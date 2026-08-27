"""Base config + hook contract for researcher-defined custom tasks (ADR-058).

A custom task is one documented ``.py`` under ``user_tasks/``: a Pydantic
``Config`` extending :class:`BaseTaskConfig` (which already composes the shared
training/data/output/device blocks — the researcher adds only task-specific
fields) and a :class:`TaskSpec` subclass with four hooks. VisionForge's generic
engine drives the loop, checkpointing, SSE and run.json around them;
the researcher never touches React, FastAPI, or the epoch loop.

Torch types appear only in annotations (``TYPE_CHECKING``): importing this
module — e.g. for schema generation — must not require the hardware extra.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import BaseModel, Field

from visionforge.utils.config import (
    CURRENT_SCHEMA_VERSION,
    DETERMINISTIC_DESCRIPTION,
    DeviceConfig,
    OutputConfig,
    PreprocessingConfig,
    SchedulerConfig,
    TransformConfig,
)

if TYPE_CHECKING:  # pragma: no cover - annotations only
    import torch
    from torch import nn


class TaskTrainingConfig(BaseModel):
    """Shared training knobs every custom task inherits."""

    epochs: int = Field(default=10, ge=1)
    batch_size: int = Field(default=32, ge=1)
    learning_rate: float = Field(default=0.001, gt=0)
    optimizer: Literal["adam", "sgd", "adamw"] = "adam"
    weight_decay: float = Field(default=0.0, ge=0)
    # Zero by default because the previous default never fired: with
    # `epochs=10` it took ten consecutive epochs without improvement inside
    # ten epochs. It read as a protection that was not there.
    early_stopping_patience: int = Field(
        default=0,
        ge=0,
        description="Épocas seguidas sem melhora antes de encerrar o treino. 0 (padrão) desliga: roda todas as épocas configuradas.",
    )
    seed: int = Field(default=42, ge=0)
    deterministic: bool = Field(default=True, description=DETERMINISTIC_DESCRIPTION)
    mixed_precision: bool = False
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)


class TaskDataConfig(BaseModel):
    """Shared dataset knobs; task-specific fields live on the Config subclass."""

    base_dir: Path
    image_size: int = Field(default=224, ge=32)
    num_workers: int = Field(default=4, ge=0)
    pin_memory: bool = True
    transforms: TransformConfig = Field(default_factory=TransformConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)


class BaseTaskConfig(BaseModel):
    """Root config a custom task's ``Config`` extends (ADR-058).

    Carries ``schema_version`` like every built-in config (ADR-039) and the
    shared blocks; the subclass adds only what is task-specific.
    """

    schema_version: int = CURRENT_SCHEMA_VERSION
    name: str = "custom_run"
    data: TaskDataConfig
    training: TaskTrainingConfig = Field(default_factory=TaskTrainingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    device: DeviceConfig = Field(default_factory=DeviceConfig)


class TaskSpec[ConfigT: BaseTaskConfig](ABC):
    """The four hooks a custom task implements (Level 1).

    The generic engine calls them in order: ``build_model`` →
    ``build_loaders`` → per-epoch ``compute_loss`` over train batches →
    ``compute_metrics`` on the val loader (best checkpoint by the task's
    primary metric). Tasks whose training is not epoch-shaped override
    :meth:`run` instead (Level 2) and own the whole loop.

    Subclass as ``TaskSpec[YourConfig]`` (and set ``Config = YourConfig``) so
    the hooks can be annotated with the task's own config type.
    """

    Config: ClassVar[type[BaseTaskConfig]] = BaseTaskConfig

    @abstractmethod
    def build_model(self, cfg: ConfigT) -> nn.Module:
        """Return the freshly-initialized model for this config."""

    @abstractmethod
    def build_loaders(self, cfg: ConfigT) -> tuple[Any, Any, Any | None]:
        """Return ``(train_loader, val_loader, test_loader_or_None)``.

        Datasets defined in the task file itself require ``num_workers=0``:
        spawned loader workers cannot re-import a path-loaded module
        (ADR-030).
        """

    @abstractmethod
    def compute_loss(self, model: nn.Module, batch: Any, cfg: ConfigT) -> torch.Tensor:
        """Return the scalar training loss for one batch (model already on device)."""

    @abstractmethod
    def compute_metrics(
        self, model: nn.Module, loader: Any, cfg: ConfigT
    ) -> dict[str, float]:
        """Evaluate ``model`` over ``loader`` and return ``{metric: value}``.

        Keys must match the metric names declared in ``@register_task``.
        """

    def run(self, cfg: ConfigT, ctx: Any) -> dict[str, float] | None:
        """Level 2 escape hatch: own the whole training loop.

        Return the final metrics dict to bypass the generic engine entirely;
        the default ``None`` means "use the Level 1 hooks". ``ctx``
        provides ``run_dir``, ``emit(event)``, ``save_checkpoint`` and
        ``write_run_json`` so a custom loop still honours the contracts.
        """
        return None

    def has_custom_run(self) -> bool:
        """True when the subclass overrides :meth:`run` (Level 2)."""
        return type(self).run is not TaskSpec.run


__all__ = [
    "BaseTaskConfig",
    "TaskDataConfig",
    "TaskTrainingConfig",
    "TaskSpec",
]
