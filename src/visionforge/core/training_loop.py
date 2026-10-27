"""Shared training lifecycle helpers with explicit task-specific inputs."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader

from visionforge.core.cancellation import CancellationToken, is_cancelled
from visionforge.core.dataset_fingerprint import fingerprint_from_config
from visionforge.core.resume import (
    ResumeState,
    capture_rng_state,
    clear_resume_state,
    require_resume_state,
    restore_rng_state,
    save_resume_state,
)
from visionforge.core.resume_provenance import validate_resume_dataset
from visionforge.core.tracking import TensorBoardLogger
from visionforge.utils.environment import capture_environment

type ParameterGroups = tuple[list[nn.Parameter], list[nn.Parameter]]


@dataclass(frozen=True)
class MetricPolicy:
    """Select the epoch metric and its checkpoint improvement direction."""

    name: str
    higher_is_better: bool = False


@dataclass
class EpochState[Epoch]:
    """Carry completed history and stopping bookkeeping between epochs."""

    history: list[Epoch]
    best_metric: float
    best_epoch: int
    patience_counter: int
    start_epoch: int


@dataclass(frozen=True)
class EpochRuntime:
    """Bind loaders and lifecycle boundaries for one epoch loop."""

    train_loader: DataLoader[Any]
    val_loader: DataLoader[Any]
    total_epochs: int
    resumed: ResumeState | None
    cancel_token: CancellationToken | None


@dataclass(frozen=True)
class CheckpointContext:
    """Bind a model's checkpoint destination and serialization adapters."""

    model: nn.Module
    model_path: Path
    save_best: Callable[
        [ThreadPoolExecutor, Future[None] | None, nn.Module, Path], Future[None]
    ]
    components: Callable[[], dict[str, Any]]


@dataclass
class CheckpointWrites:
    """Track the serial checkpoint writer and its pending best-model save."""

    pool: ThreadPoolExecutor
    future: Future[None] | None = None


@dataclass(frozen=True)
class EpochCallbacks[Epoch]:
    """Provide task computation, reporting and the early-stop boundary."""

    compute: Callable[[int], Epoch]
    report: Callable[[Epoch], None]
    early_stop: Callable[[int], bool]


@dataclass(frozen=True)
class FitOptions:
    """Carry caller-owned optimization, progress and continuation controls."""

    optimizer: torch.optim.Optimizer | None
    progress_callback: Callable[[dict[str, Any]], None] | None
    cancel_token: CancellationToken | None
    resume_dir: Path | None


@dataclass(frozen=True)
class FitServices:
    """Provide the task's seeding, logging and checkpoint implementations."""

    seed: Callable[..., None]
    tensorboard_factory: Callable[[Path], TensorBoardLogger]
    save_best: Callable[
        [ThreadPoolExecutor, Future[None] | None, nn.Module, Path], Future[None]
    ]


@dataclass(frozen=True)
class FitResultAdapter[Result]:
    """Describe task-specific ranking and final training-result construction."""

    result_type: Callable[..., Result]
    metric: MetricPolicy
    best_field: str
    log_format: str
    ensure_checkpoint: Callable[..., tuple[int, float]] | None = None


def build_scheduler(
    optimizer: torch.optim.Optimizer, cfg_training: Any
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Build the configured scheduler using validation loss for plateau mode."""
    sched_cfg = cfg_training.scheduler
    if sched_cfg.kind == "none":
        return None
    if sched_cfg.kind == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg_training.epochs
        )
    if sched_cfg.kind == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=sched_cfg.step_size, gamma=sched_cfg.gamma
        )
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=sched_cfg.patience,
        factor=sched_cfg.factor,
        min_lr=sched_cfg.min_lr,
    )


def make_run_dir(models_dir: Path, name: str) -> Path:
    """Create a timestamped run directory without overwriting an existing run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = models_dir / name / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def resolve_run_dir(
    resumed: ResumeState | None,
    resume_dir: Path | None,
    create_run_dir: Callable[[], Path],
) -> Path:
    """Reuse the resume directory before creating output artifacts."""
    if resumed is not None and resume_dir is not None:
        return resume_dir
    return create_run_dir()


def early_stopping_reached(patience_counter: int, patience: int) -> bool:
    """Check the stopping boundary while keeping zero patience disabled."""
    return patience > 0 and patience_counter >= patience


def step_scheduler(scheduler: Any, val_loss: float) -> None:
    """Advance plateau schedulers with loss and other schedulers by epoch."""
    if scheduler is None:
        return
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(val_loss)
        return
    scheduler.step()


def scheduler_state_dict(scheduler: Any) -> dict[str, Any] | None:
    """Capture scheduler state when scheduling is enabled."""
    if scheduler is None:
        return None
    state: dict[str, Any] = scheduler.state_dict()
    return state


def restore_training_state[Epoch](
    resumed: ResumeState | None,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    run_dir: Path,
    total_epochs: int,
    result_type: Callable[..., Epoch],
    initial_best: float,
    scaler: Any = None,
) -> tuple[list[Epoch], float, int, int, int]:
    """Restore optimization state and task-specific history before loader RNG."""
    if resumed is None:
        return [], initial_best, 0, 0, 1
    model.load_state_dict(resumed.model)
    optimizer.load_state_dict(resumed.optimizer)
    if scheduler is not None and resumed.scheduler is not None:
        scheduler.load_state_dict(resumed.scheduler)
    if scaler is not None and resumed.scaler is not None:
        scaler.load_state_dict(resumed.scaler)
    history = [result_type(**h) for h in resumed.history]
    start_epoch = resumed.epoch + 1
    logger.info(
        "Resuming {} at epoch {} of {}.", run_dir.name, start_epoch, total_epochs
    )
    return (
        history,
        resumed.best_metric,
        resumed.best_epoch,
        resumed.patience_counter,
        start_epoch,
    )


def start_training(
    data_module: Any,
    progress_callback: Callable[[dict[str, Any]], None] | None,
    total_epochs: int,
    device_label: str,
) -> tuple[Any, Any, dict[str, int]]:
    """Emit the start event and build each persistent loader exactly once."""
    if progress_callback is not None:
        progress_callback(
            {"event": "start", "total_epochs": total_epochs, "device": device_label}
        )
    train_loader = data_module.train_loader()
    val_loader = data_module.val_loader()
    workers = {
        name: loader.num_workers
        for name, loader in (("train", train_loader), ("val", val_loader))
        if hasattr(loader, "num_workers")
    }
    return train_loader, val_loader, workers


def checkpoint_pool(
    resumed: ResumeState | None,
    train_loader: Any,
    val_loader: Any,
) -> ThreadPoolExecutor:
    """Create the save pool before restoring randomness consumed by loaders."""
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ckpt")
    if resumed is not None:
        restore_rng_state(resumed.rng_state, train_loader, val_loader)
    return pool


def cancelled_at_epoch(cancel_token: CancellationToken | None, epoch: int) -> bool:
    """Log cancellation at the safe boundary between completed epochs."""
    if not is_cancelled(cancel_token):
        return False
    logger.info("Run cancelled at epoch {}; keeping the best checkpoint so far.", epoch)
    return True


def checkpoint_components(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any = None,
    *,
    unwrap_model: bool = False,
) -> dict[str, Any]:
    """Snapshot optimization components preserving the task's model wrapping."""
    core = (
        model.module if unwrap_model and isinstance(model, nn.DataParallel) else model
    )
    return {
        "model": core.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler_state_dict(scheduler),
        "scaler": scaler.state_dict() if scaler is not None else None,
    }


def save_epoch_checkpoint(
    metric: float,
    epoch: int,
    state: EpochState[Any],
    checkpoint: CheckpointContext,
    writes: CheckpointWrites,
    runtime: EpochRuntime,
    policy: MetricPolicy,
) -> None:
    """Save an improved checkpoint and snapshot every completed epoch for resume."""
    improved = (
        metric > state.best_metric
        if policy.higher_is_better
        else metric < state.best_metric
    )
    if improved:
        state.best_metric, state.best_epoch, state.patience_counter = metric, epoch, 0
        writes.future = checkpoint.save_best(
            writes.pool, writes.future, checkpoint.model, checkpoint.model_path
        )
    else:
        state.patience_counter += 1
    save_resume_state(
        checkpoint.model_path.parent,
        ResumeState(
            epoch=epoch,
            **checkpoint.components(),
            best_metric=state.best_metric,
            best_epoch=state.best_epoch,
            patience_counter=state.patience_counter,
            history=[asdict(h) for h in state.history],
            rng_state=capture_rng_state(runtime.train_loader, runtime.val_loader),
        ),
    )


def finish_checkpoint_writes(
    run_dir: Path,
    cancel_token: CancellationToken | None,
    save_pool: ThreadPoolExecutor,
    save_future: Future[None] | None,
) -> None:
    """Clear completed resume state and wait for pending checkpoint writes."""
    if not is_cancelled(cancel_token):
        clear_resume_state(run_dir)
    if save_future is not None:
        save_future.result()
    save_pool.shutdown(wait=False)


def close_training(
    tb: TensorBoardLogger,
    device: torch.device,
    progress_callback: Callable[[dict[str, Any]], None] | None,
    total_epochs: int,
) -> None:
    """Close logging, release cached device memory and emit the end event."""
    tb.close()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if progress_callback is not None:
        progress_callback({"event": "end", "total_epochs": total_epochs})


def report_epoch(
    result: Any,
    tb: TensorBoardLogger,
    progress_callback: Callable[[dict[str, Any]], None] | None,
    t0: float,
    total_epochs: int,
    scalar_fields: dict[str, str],
    event_fields: tuple[str, ...],
    log_format: str,
    log_fields: tuple[str, ...],
) -> None:
    """Report an epoch using the task's scalar names, event fields and log format."""
    tb.log_scalars(
        result.epoch,
        {name: getattr(result, field) for name, field in scalar_fields.items()},
    )
    logger.info(
        log_format,
        result.epoch,
        total_epochs,
        *(getattr(result, field) for field in log_fields),
    )
    if progress_callback is not None:
        progress_callback(
            {
                "event": "epoch_end",
                "epoch": result.epoch,
                "total_epochs": total_epochs,
                **{field: getattr(result, field) for field in event_fields},
                "elapsed_s": round(time.monotonic() - t0, 3),
            }
        )


def run_json_header(
    config: Any,
    run_dir: Path,
    stopped: bool,
    device_used: str,
    loader_workers: dict[str, int],
) -> dict[str, Any]:
    """Build the common ADR-013 metadata without changing task-specific fields."""
    return {
        "id": f"{config.name}_{run_dir.name}",
        "experiment": config.name,
        "timestamp": datetime.now().isoformat(),
        "status": "completed",
        "stopped": stopped,
        "device_used": device_used,
        "environment": capture_environment(),
        "runtime": {"loader_workers": loader_workers},
        "dataset_fingerprint": fingerprint_from_config(config),
        "run_dir": str(run_dir.resolve()),
    }


def build_optimizer(
    model: nn.Module,
    cfg_training: Any,
    transfer_learning: Any,
    split_params: Callable[[nn.Module], tuple[list[nn.Parameter], list[nn.Parameter]]],
) -> torch.optim.Optimizer:
    """Build the optimizer with the task's transfer-learning parameter split."""
    builders: dict[str, Callable[..., torch.optim.Optimizer]] = {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
        "adamw": torch.optim.AdamW,
    }
    builder = builders[cfg_training.optimizer]
    if transfer_learning is None:
        return builder(
            model.parameters(),
            lr=cfg_training.learning_rate,
            weight_decay=cfg_training.weight_decay,
        )
    if transfer_learning.mode == "feature_extraction":
        return builder(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg_training.learning_rate,
            weight_decay=cfg_training.weight_decay,
        )
    head, backbone = split_params(model)
    return builder(
        [
            {
                "params": backbone,
                "lr": cfg_training.learning_rate
                * transfer_learning.backbone_lr_multiplier,
                "weight_decay": cfg_training.weight_decay,
            },
            {
                "params": head,
                "lr": cfg_training.learning_rate,
                "weight_decay": cfg_training.weight_decay,
            },
        ]
    )


def apply_transfer_learning(
    model: nn.Module,
    transfer_learning: Any,
    split_params: Callable[[nn.Module], tuple[list[nn.Parameter], list[nn.Parameter]]],
) -> None:
    """Freeze the backbone using the task-specific head parameter split."""
    if transfer_learning is None or transfer_learning.mode != "feature_extraction":
        return
    head, backbone = split_params(model)
    for p in backbone:
        p.requires_grad = False
    for p in head:
        p.requires_grad = True


def prepare_model(
    model: nn.Module, device: torch.device, dp_ids: list[int]
) -> nn.Module:
    """Move the model to its device and apply the configured DataParallel wrapper."""
    model = model.to(device)
    if dp_ids:
        model = nn.DataParallel(model, device_ids=dp_ids)
    return model


def prepare_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    prepare: Callable[[nn.Module], nn.Module],
    transfer: Callable[[nn.Module], None],
    optimizer_builder: Callable[[nn.Module], torch.optim.Optimizer],
    criterion_builder: Callable[[], nn.Module],
    scheduler_builder: Callable[[torch.optim.Optimizer], Any],
) -> tuple[nn.Module, torch.optim.Optimizer, nn.Module, Any]:
    """Prepare task-specific optimization tools in their original order."""
    model = prepare(model)
    transfer(model)
    optimizer = optimizer if optimizer is not None else optimizer_builder(model)
    criterion = criterion_builder()
    scheduler = scheduler_builder(optimizer)
    return model, optimizer, criterion, scheduler


def load_resume_state(resume_dir: Path | None, config: Any) -> ResumeState | None:
    """Load resume state and validate its dataset before initializing training."""
    resumed = require_resume_state(resume_dir) if resume_dir is not None else None
    if resume_dir is not None:
        validate_resume_dataset(resume_dir, config)
    return resumed


def run_epochs[Epoch](
    state: EpochState[Epoch],
    runtime: EpochRuntime,
    checkpoint: CheckpointContext,
    callbacks: EpochCallbacks[Epoch],
    metric: MetricPolicy,
) -> tuple[CheckpointWrites, bool]:
    """Run epochs with shared stopping boundaries and checkpoint bookkeeping."""
    writes = CheckpointWrites(
        checkpoint_pool(runtime.resumed, runtime.train_loader, runtime.val_loader)
    )
    stopped = False
    for epoch in range(state.start_epoch, runtime.total_epochs + 1):
        # Resuming at a stopping boundary must not train an extra epoch.
        if callbacks.early_stop(state.patience_counter):
            break
        if cancelled_at_epoch(runtime.cancel_token, epoch):
            stopped = True
            break
        result = callbacks.compute(epoch)
        state.history.append(result)
        callbacks.report(result)
        save_epoch_checkpoint(
            getattr(result, metric.name),
            epoch,
            state,
            checkpoint,
            writes,
            runtime,
            metric,
        )
        if callbacks.early_stop(state.patience_counter):
            logger.info("Early stopping at epoch {}.", epoch)
            break
    return writes, stopped


def seed_training(
    cfg_training: Any,
    seed: Callable[..., None],
    device_label: str,
    log_format: str,
) -> None:
    """Seed training before model preparation and report the resolved device."""
    seed(cfg_training.seed, deterministic=cfg_training.deterministic)
    logger.info(log_format, device_label)


def train_epoch(
    model: nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    *,
    outputs_transform: Callable[[Any], torch.Tensor] | None = None,
    validate_targets: Callable[[torch.Tensor], None] | None = None,
) -> float:
    """Train a non-AMP epoch using explicit task output and target adapters."""
    model.train()
    total_loss = 0.0
    for inputs, targets in loader:
        if validate_targets is not None:
            validate_targets(targets)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        if outputs_transform is not None:
            outputs = outputs_transform(outputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    n = len(loader)
    return total_loss / n if n > 0 else 0.0


def fit_training[Result](
    trainer: Any,
    model: nn.Module,
    data_module: Any,
    options: FitOptions,
    services: FitServices,
    adapter: FitResultAdapter[Result],
) -> Result:
    """Fit a non-AMP task with explicit metric, checkpoint and result adapters."""
    optimizer = options.optimizer
    progress_callback = options.progress_callback
    cancel_token = options.cancel_token
    resume_dir = options.resume_dir
    resumed = load_resume_state(resume_dir, trainer._config)
    cfg = trainer._config.training
    seed_training(
        cfg,
        services.seed,
        trainer._device_label,
        adapter.log_format,
    )

    model, optimizer, criterion, scheduler = prepare_training(
        model,
        optimizer,
        trainer._prepare_model,
        trainer._apply_transfer_learning,
        trainer._build_optimizer,
        trainer._build_criterion,
        trainer._build_scheduler,
    )
    # Resolved before anything is created: a resumed run must continue in its
    # own directory, or it leaves an empty one behind and writes its
    # TensorBoard events somewhere nothing will look.
    run_dir = trainer._resolve_run_dir(resumed, resume_dir)
    model_path = run_dir / "best_model.pth"
    tb = services.tensorboard_factory(run_dir / "tensorboard")

    history, best_val_loss, best_epoch, patience_counter, start_epoch = (
        trainer._restore_training_state(resumed, model, optimizer, scheduler, run_dir)
    )
    t0 = time.monotonic()

    train_loader, val_loader, trainer._loader_workers = start_training(
        data_module, progress_callback, cfg.epochs, trainer._device_label
    )
    state = EpochState(
        history, best_val_loss, best_epoch, patience_counter, start_epoch
    )
    writes, stopped = run_epochs(
        state,
        EpochRuntime(train_loader, val_loader, cfg.epochs, resumed, cancel_token),
        CheckpointContext(
            model,
            model_path,
            services.save_best,
            lambda: checkpoint_components(model, optimizer, scheduler),
        ),
        EpochCallbacks(
            compute=lambda epoch: trainer._compute_epoch_result(
                epoch, model, train_loader, val_loader, optimizer, criterion, scheduler
            ),
            report=lambda result: trainer._report_epoch(
                result, tb, progress_callback, t0, cfg.epochs
            ),
            early_stop=lambda counter: trainer._early_stopping_reached(
                counter, cfg.early_stopping_patience
            ),
        ),
        adapter.metric,
    )
    best_val_loss, best_epoch = state.best_metric, state.best_epoch

    trainer._finish_checkpoint_writes(run_dir, cancel_token, writes.pool, writes.future)

    health = trainer._training_warnings(history)

    if adapter.ensure_checkpoint is not None:
        best_epoch, best_val_loss = adapter.ensure_checkpoint(
            model, model_path, history, best_epoch, best_val_loss
        )
    train_result = adapter.result_type(
        stopped=stopped,
        best_epoch=best_epoch,
        **{adapter.best_field: best_val_loss},
        total_epochs=len(history),
        device_used=trainer._device_label,
        history=history,
        model_path=model_path,
        warnings=health,
    )
    trainer._write_run_json(run_dir, train_result)
    trainer._close_training(tb, progress_callback, len(history))

    return train_result


class TrainingOutput:
    """Common device label and output paths for standalone trainers."""

    _config: Any
    _device_label: str

    @property
    def device_label(self) -> str:
        """Human-readable label of the device actually used (or fallback)."""
        return self._device_label

    def _make_run_dir(self) -> Path:
        """Create the configured timestamped output directory."""
        return make_run_dir(self._config.output.models_dir, self._config.name)


class GradientTrainerMethods(TrainingOutput):
    """Shared infrastructure for non-AMP gradient trainers."""

    _device: torch.device
    _dp_ids: list[int]
    _epoch_result_type: Callable[..., Any]
    _initial_best: float
    _epoch_fields: tuple[str, ...]
    _train_epoch: Any
    _eval_epoch: Any

    _early_stopping_reached = staticmethod(early_stopping_reached)
    _step_scheduler = staticmethod(step_scheduler)
    _scheduler_state_dict = staticmethod(scheduler_state_dict)
    _finish_checkpoint_writes = staticmethod(finish_checkpoint_writes)

    def _resolve_run_dir(
        self, resumed: ResumeState | None, resume_dir: Path | None
    ) -> Path:
        """Reuse the resume directory before creating output artifacts."""
        return resolve_run_dir(resumed, resume_dir, self._make_run_dir)

    def _restore_training_state(
        self,
        resumed: ResumeState | None,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        run_dir: Path,
    ) -> tuple[list[Any], float, int, int, int]:
        """Restore optimization state and task-specific epoch history."""
        return restore_training_state(
            resumed,
            model,
            optimizer,
            scheduler,
            run_dir,
            self._config.training.epochs,
            self._epoch_result_type,
            self._initial_best,
        )

    def _close_training(
        self,
        tb: TensorBoardLogger,
        progress_callback: Callable[[dict[str, Any]], None] | None,
        total_epochs: int,
    ) -> None:
        """Close logging and emit the final progress event."""
        close_training(tb, self._device, progress_callback, total_epochs)

    def _prepare_model(self, model: nn.Module) -> nn.Module:
        """Move the model to its device and apply DataParallel when configured."""
        return prepare_model(model, self._device, self._dp_ids)

    def _apply_transfer_learning(self, model: nn.Module) -> None:
        """Freeze the backbone for feature extraction."""
        apply_transfer_learning(
            model, self._config.transfer_learning, self._split_named_params
        )

    def _build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Build the optimizer using the task-specific parameter split."""
        return build_optimizer(
            model,
            self._config.training,
            self._config.transfer_learning,
            self._split_named_params,
        )

    def _build_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler.LRScheduler | None:
        """Build the configured learning-rate scheduler."""
        return build_scheduler(optimizer, self._config.training)

    @staticmethod
    def _split_named_params(model: nn.Module) -> ParameterGroups:
        """Require the task to define its head and backbone split."""
        raise NotImplementedError

    def _compute_epoch_result(
        self,
        epoch: int,
        model: nn.Module,
        train_loader: Any,
        val_loader: Any,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scheduler: Any,
    ) -> Any:
        """Compute task metrics and advance scheduling in the original order."""
        train_loss = self._train_epoch(model, train_loader, optimizer, criterion)
        val_loss, *values = self._eval_epoch(model, val_loader, criterion)
        self._step_scheduler(scheduler, val_loss)
        return self._epoch_result_type(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            **dict(zip(self._epoch_fields, values, strict=True)),
        )


def run_json_tail(result: Any, graphics: list[str] | None = None) -> dict[str, Any]:
    """Build common run artifacts and health metadata without inventing weights."""
    return {
        "artifacts": {
            # An untrained run must not advertise a checkpoint (ADR-111).
            "model": str(result.model_path) if result.model_path.is_file() else None,
            "graphics": graphics if graphics is not None else [],
            "report": None,
        },
        "tests": [],
        "warnings": result.warnings,
    }


def write_task_run_json(
    config: Any,
    run_dir: Path,
    result: Any,
    loader_workers: dict[str, int],
    *,
    best_metric: str,
    epoch_metrics: dict[str, str],
    history_fields: tuple[str, ...],
    mask_empty_best: bool = True,
) -> None:
    """Write standalone task metadata with its original metric and history order."""
    best = next((r for r in result.history if r.epoch == result.best_epoch), None)
    best_value = getattr(result, best_metric)
    run_json = {
        **run_json_header(
            config, run_dir, result.stopped, result.device_used, loader_workers
        ),
        "config": config.model_dump(mode="json"),
        "metrics": {
            # Empty runs must not turn sentinels into measured metrics (ADR-111).
            best_metric: best_value
            if result.total_epochs or not mask_empty_best
            else None,
            "best_epoch": result.best_epoch,
            "total_epochs": result.total_epochs,
            **{
                name: getattr(best, field) if best else None
                for name, field in epoch_metrics.items()
            },
        },
        "history": [
            {field: getattr(r, field) for field in history_fields}
            for r in result.history
        ],
        **run_json_tail(result),
    }
    (run_dir / "run.json").write_text(json.dumps(run_json, indent=2), encoding="utf-8")


def training_warnings(
    history: Any,
    summarize: Callable[..., list[dict[str, str]]],
    stagnant_loss: Callable[..., Any],
    additional: Callable[[], list[Any]] | None = None,
) -> list[dict[str, str]]:
    """Summarize and log task health checks in their original order."""
    checks = [stagnant_loss([h.train_loss for h in history])]
    if additional is not None:
        checks.extend(additional())
    health = summarize(checks)
    for warning in health:
        logger.warning("{}", warning["message"])
    return health
