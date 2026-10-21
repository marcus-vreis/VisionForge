"""Everything a run needs to pick up where it stopped.

Runs die. A page file too small for the DataLoader workers (ADR-081), a machine
that reboots, a stop the researcher later regrets (ADR-088) — and until now all
of them cost the whole run, because the only thing on disk was the best weights.
Weights alone cannot continue: an optimizer that has forgotten its momentum and
a scheduler that has forgotten its step restart training somewhere the loss
curve never was.

**The deliverable checkpoint is deliberately not the resume file.** `best.pth`
stays a bare `state_dict`, because five things load it — evaluation, Grad-CAM,
batch prediction, ONNX export and the per-model test — all with
`weights_only=True`. Turning it into a dict of dicts would break every one of
them to serve a feature none of them use. So the training state lives beside it
in `resume.pt`, which is disposable: deleting it costs the ability to continue,
nothing else.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from loguru import logger

RESUME_FILENAME = "resume.pt"

# Version 1 remains readable, but cannot reproduce the random trajectory.
_FORMAT_VERSION = 2


@dataclass
class ResumeState:
    """The training state at the end of a completed epoch."""

    epoch: int  # last epoch that finished
    model: dict[str, Any]
    optimizer: dict[str, Any]
    scheduler: dict[str, Any] | None
    scaler: dict[str, Any] | None
    best_metric: float
    best_epoch: int
    patience_counter: int
    history: list[dict[str, Any]]
    rng_state: dict[str, Any] | None = None


def _loader_generators(loaders: tuple[Any, ...]) -> dict[str, torch.Generator]:
    generators = {}
    for index, loader in enumerate(loaders):
        for name, owner in (
            ("loader", loader),
            ("sampler", getattr(loader, "sampler", None)),
            (
                "batch_sampler",
                getattr(getattr(loader, "batch_sampler", None), "sampler", None),
            ),
        ):
            generator = getattr(owner, "generator", None)
            if isinstance(generator, torch.Generator):
                generators[f"{index}.{name}"] = generator
    return generators


def capture_rng_state(*loaders: Any) -> dict[str, Any]:
    """Capture global RNGs and explicit loader generators without initializing CUDA."""
    numpy_state = cast(tuple[Any, ...], np.random.get_state())
    return {
        "python": random.getstate(),
        "numpy": (
            numpy_state[0],
            numpy_state[1].tolist(),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        ),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
        "loaders": {
            name: generator.get_state()
            for name, generator in _loader_generators(loaders).items()
        },
    }


def _validate_rng_state(state: dict[str, Any]) -> None:
    # Validate with isolated generators so a corrupt checkpoint cannot partly
    # reset the process RNGs before loading is refused.
    random.Random().setstate(state["python"])
    np.random.RandomState().set_state(state["numpy"])
    torch.Generator().set_state(state["torch"])
    for value in state["loaders"].values():
        torch.Generator().set_state(value)
    if not isinstance(state["cuda"], list) or any(
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.uint8
        or value.ndim != 1
        for value in state["cuda"]
    ):
        raise ValueError("Invalid CUDA RNG state")


def restore_rng_state(state: dict[str, Any] | None, *loaders: Any) -> None:
    """Restore an epoch boundary after model, optimizer and loader setup."""
    if state is None:
        return
    _validate_rng_state(state)
    generators = _loader_generators(loaders)
    if generators.keys() != state["loaders"].keys():
        raise ValueError("Resume loader generators differ from the saved configuration")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    for name, generator in generators.items():
        generator.set_state(state["loaders"][name])
    if state["cuda"]:
        if (
            torch.cuda.is_available()
            and len(state["cuda"]) == torch.cuda.device_count()
        ):
            torch.cuda.set_rng_state_all(state["cuda"])
        else:
            logger.warning(
                "CUDA RNG state cannot be restored on this device configuration; exact resume is unavailable."
            )
    if any(getattr(loader, "num_workers", 0) > 0 for loader in loaders):
        logger.warning(
            "Resume does not restore DataLoader worker RNGs or prefetch queues; exact resume requires num_workers=0."
        )


def resume_path(run_dir: Path) -> Path:
    """Where a run keeps its resume state."""
    return run_dir / RESUME_FILENAME


def save_resume_state(run_dir: Path, state: ResumeState) -> None:
    """Write the state atomically, so a crash mid-write cannot poison it.

    A half-written resume file is the one failure this feature must not add: it
    would be found on the next attempt, refuse to load, and leave the researcher
    believing resume is broken rather than that the run died.
    """
    target = resume_path(run_dir)
    tmp = target.with_suffix(".pt.tmp")
    payload = {
        "format_version": _FORMAT_VERSION,
        "epoch": state.epoch,
        "model": state.model,
        "optimizer": state.optimizer,
        "scheduler": state.scheduler,
        "scaler": state.scaler,
        "best_metric": state.best_metric,
        "best_epoch": state.best_epoch,
        "patience_counter": state.patience_counter,
        "history": state.history,
        "rng_state": state.rng_state
        if state.rng_state is not None
        else capture_rng_state(),
    }
    try:
        torch.save(payload, tmp)
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


def load_resume_state(run_dir: Path) -> ResumeState | None:
    """Read the state, or None when there is nothing usable to continue from.

    Never raises: every failure here means "start fresh", and a run that cannot
    be continued is a smaller problem than one that refuses to start.
    """
    target = resume_path(run_dir)
    if not target.is_file():
        return None
    try:
        payload = torch.load(target, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - unreadable state is not fatal
        logger.warning("Resume state at {} could not be read: {}", target, exc)
        return None
    if not isinstance(payload, dict):
        logger.warning("Resume state at {} is not a state dict; ignoring.", target)
        return None
    version = payload.get("format_version")
    if version not in (1, _FORMAT_VERSION):
        logger.warning(
            "Resume state at {} is format {}, this build writes {}; ignoring.",
            target,
            version,
            _FORMAT_VERSION,
        )
        return None
    try:
        rng_state = payload["rng_state"] if version == _FORMAT_VERSION else None
        if version == _FORMAT_VERSION:
            if not isinstance(rng_state, dict):
                raise ValueError("Missing RNG snapshot")
            _validate_rng_state(rng_state)
        state = ResumeState(
            epoch=int(payload["epoch"]),
            model=payload["model"],
            optimizer=payload["optimizer"],
            scheduler=payload.get("scheduler"),
            scaler=payload.get("scaler"),
            best_metric=float(payload["best_metric"]),
            best_epoch=int(payload["best_epoch"]),
            patience_counter=int(payload.get("patience_counter", 0)),
            history=list(payload.get("history", [])),
            rng_state=rng_state,
        )
        if version == 1:
            logger.warning(
                "Legacy resume state at {} has no RNG snapshot; continuing without exact reproducibility.",
                target,
            )
        return state
    except (KeyError, TypeError, ValueError, RuntimeError, AttributeError) as exc:
        logger.warning("Resume state at {} is missing fields: {}", target, exc)
        return None


def clear_resume_state(run_dir: Path) -> None:
    """Drop the resume file once the run has nothing left to continue.

    Kept for a run that stopped early and removed for one that reached its last
    epoch, so the presence of the file is itself the answer to "can this be
    resumed".
    """
    try:
        resume_path(run_dir).unlink(missing_ok=True)
    except OSError as exc:  # noqa: BLE001 - a leftover file is harmless
        logger.debug("Could not remove resume state: {}", exc)


def discard_resume_state(run_dir: Path) -> None:
    """Make a run not continuable, whichever loop owns its state.

    The in-house trainers resume from `resume.pt`; Ultralytics from its own
    `weights/last.pt` (ADR-093). `best.pt` and every other artifact stay.
    """
    clear_resume_state(run_dir)
    try:
        (run_dir / "weights" / "last.pt").unlink(missing_ok=True)
    except OSError as exc:  # noqa: BLE001 - a leftover file is harmless
        logger.debug("Could not remove Ultralytics resume state: {}", exc)


def can_resume(run_dir: Path, configured_epochs: int) -> bool:
    """Whether this run stopped before its last epoch and left usable state."""
    state = load_resume_state(run_dir)
    return state is not None and state.epoch < configured_epochs


__all__ = [
    "RESUME_FILENAME",
    "ResumeState",
    "can_resume",
    "capture_rng_state",
    "clear_resume_state",
    "discard_resume_state",
    "load_resume_state",
    "resume_path",
    "restore_rng_state",
    "save_resume_state",
]
