"""Best-effort TensorBoard logging (ADR-054).

``TensorBoardLogger`` writes per-epoch scalars to ``<run_dir>/tensorboard/``.
TensorBoard ships with the package (ADR-106), so this is on by default; read the
scalars with ``tensorboard --logdir outputs/models``. Every method stays a no-op
when the import fails, so a damaged install degrades instead of failing a run —
the ``SummaryWriter`` import is lazy and failure-tolerant.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger


class TensorBoardLogger:
    """Logs scalars to a TensorBoard ``SummaryWriter``; no-op if unavailable."""

    def __init__(self, log_dir: Path | str) -> None:
        self._writer: Any | None = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            self._writer = SummaryWriter(log_dir=str(log_dir))
        except Exception as exc:  # noqa: BLE001 — missing extra must not break training
            logger.debug("TensorBoard logging disabled: {}", exc)
            self._writer = None

    @property
    def enabled(self) -> bool:
        """True when a writer was created (the ``tensorboard`` extra is present)."""
        return self._writer is not None

    def log_scalars(self, step: int, scalars: dict[str, float | None]) -> None:
        """Write ``{tag: value}`` at ``step`` (epoch); skip ``None`` values. No-op if disabled."""
        if self._writer is None:
            return
        for tag, value in scalars.items():
            if value is not None:
                self._writer.add_scalar(tag, float(value), step)

    def close(self) -> None:
        """Flush and close the writer (idempotent)."""
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
            self._writer = None


__all__ = ["TensorBoardLogger"]
