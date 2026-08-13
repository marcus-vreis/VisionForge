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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from loguru import logger

RESUME_FILENAME = "resume.pt"

# Bumped when the fields below change shape. A file from a different version is
# refused rather than partially applied: restoring half a training state is
# worse than starting over, because the run would look continued and not be.
_FORMAT_VERSION = 1


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
    }
    torch.save(payload, tmp)
    os.replace(tmp, target)


def load_resume_state(run_dir: Path) -> ResumeState | None:
    """Read the state, or None when there is nothing usable to continue from.

    Never raises: every failure here means "start fresh", and a run that cannot
    be continued is a smaller problem than one that refuses to start.
    """
    target = resume_path(run_dir)
    if not target.is_file():
        return None
    try:
        payload = torch.load(target, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 - unreadable state is not fatal
        logger.warning("Resume state at {} could not be read: {}", target, exc)
        return None
    if not isinstance(payload, dict):
        logger.warning("Resume state at {} is not a state dict; ignoring.", target)
        return None
    version = payload.get("format_version")
    if version != _FORMAT_VERSION:
        logger.warning(
            "Resume state at {} is format {}, this build writes {}; ignoring.",
            target,
            version,
            _FORMAT_VERSION,
        )
        return None
    try:
        return ResumeState(
            epoch=int(payload["epoch"]),
            model=payload["model"],
            optimizer=payload["optimizer"],
            scheduler=payload.get("scheduler"),
            scaler=payload.get("scaler"),
            best_metric=float(payload["best_metric"]),
            best_epoch=int(payload["best_epoch"]),
            patience_counter=int(payload.get("patience_counter", 0)),
            history=list(payload.get("history", [])),
        )
    except (KeyError, TypeError, ValueError) as exc:
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


def can_resume(run_dir: Path, configured_epochs: int) -> bool:
    """Whether this run stopped before its last epoch and left usable state."""
    state = load_resume_state(run_dir)
    return state is not None and state.epoch < configured_epochs


__all__ = [
    "RESUME_FILENAME",
    "ResumeState",
    "can_resume",
    "clear_resume_state",
    "load_resume_state",
    "resume_path",
    "save_resume_state",
]
