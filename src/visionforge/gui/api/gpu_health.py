"""Whether this server's CUDA context still works.

A device-side assert inside a kernel (an out-of-range class id in a loss, for one)
poisons the CUDA context of the whole process: every later CUDA call raises, and
the trainers seed CUDA even for a CPU job, so every later job fails within a second
until the server restarts. Nothing can repair the context from inside the process;
the GUI can only stop accepting jobs and say why.
"""

from __future__ import annotations

from loguru import logger

_CUDA_ERROR_MARKERS = ("CUDA error", "device-side assert", "cudaError")


def cuda_context_broken() -> bool:
    """True when CUDA was started in this process and no longer accepts work.

    Never starts CUDA itself: a server that has not trained on the GPU yet has no
    context to be broken. The probe is a one-element allocation, which raises at
    once on a poisoned context and does not wait for a running job's kernels.
    """
    import torch

    if not torch.cuda.is_initialized():
        return False
    try:
        torch.zeros(1, device="cuda")
    except RuntimeError as exc:
        if any(marker in str(exc) for marker in _CUDA_ERROR_MARKERS):
            logger.error("The CUDA context of this server is broken: {}", exc)
            return True
        raise
    return False


__all__ = ["cuda_context_broken"]
