from __future__ import annotations

from dataclasses import dataclass

from loguru import logger


@dataclass(frozen=True, slots=True)
class CUDAInfo:
    """Immutable snapshot of the CUDA environment at query time."""

    available: bool = False
    device_count: int = 0
    current_device: int | None = None
    device_name: str | None = None
    cuda_version: str | None = None


def check_cuda() -> CUDAInfo:
    """Probe the runtime for CUDA/GPU support.

    Returns:
        CUDAInfo snapshot. Never raises — returns CUDAInfo(available=False) on any error.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return CUDAInfo(available=False)

        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        cuda_version = torch.version.cuda

        return CUDAInfo(
            available=True,
            device_count=device_count,
            current_device=current_device,
            device_name=device_name,
            cuda_version=cuda_version,
        )
    except Exception:
        return CUDAInfo(available=False)


def log_cuda_status() -> None:
    """
    Log a one-line CUDA diagnostic.

    Emits ``INFO`` when a GPU is found, ``WARNING`` otherwise.
    """
    info = check_cuda()

    if info.available:
        logger.info(
            "CUDA available — {} (device {}/{}, CUDA {})",
            info.device_name,
            (info.current_device or 0) + 1,
            info.device_count,
            info.cuda_version,
        )
    else:
        logger.warning("CUDA not available — training will fall back to CPU")


__all__ = ["CUDAInfo", "check_cuda", "log_cuda_status"]
