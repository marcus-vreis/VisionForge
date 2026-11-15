from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger


@dataclass(frozen=True, slots=True)
class GPUDevice:
    """Per-GPU details exposed to the GUI for device selection."""

    index: int
    name: str
    total_memory_mb: int
    compute_capability: str | None = None


@dataclass(frozen=True, slots=True)
class CUDAInfo:
    """Immutable snapshot of the CUDA environment at query time."""

    available: bool = False
    device_count: int = 0
    current_device: int | None = None
    device_name: str | None = None
    cuda_version: str | None = None
    devices: tuple[GPUDevice, ...] = field(default_factory=tuple)


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

        devices: list[GPUDevice] = []
        for i in range(device_count):
            try:
                props = torch.cuda.get_device_properties(i)
                total_mb = int(props.total_memory / (1024 * 1024))
                cap = f"{props.major}.{props.minor}"
                devices.append(
                    GPUDevice(
                        index=i,
                        name=torch.cuda.get_device_name(i),
                        total_memory_mb=total_mb,
                        compute_capability=cap,
                    )
                )
            except Exception:
                devices.append(
                    GPUDevice(
                        index=i,
                        name=torch.cuda.get_device_name(i),
                        total_memory_mb=0,
                    )
                )

        return CUDAInfo(
            available=True,
            device_count=device_count,
            current_device=current_device,
            device_name=device_name,
            cuda_version=cuda_version,
            devices=tuple(devices),
        )
    except Exception:
        return CUDAInfo(available=False)


def mps_available() -> bool:
    """Whether PyTorch can use Apple's Metal GPU (MPS). Never raises (ADR-192).

    Separate from :func:`check_cuda` because it answers a different question
    with a different shape: there is one MPS device or none, with no index,
    memory figure or driver version to report. Only a real ``True`` counts, so
    a mocked ``torch`` (a ``MagicMock`` is truthy) never reads as a Mac.
    """
    try:
        import torch

        return torch.backends.mps.is_available() is True
    except Exception:
        return False


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


__all__ = ["CUDAInfo", "GPUDevice", "check_cuda", "log_cuda_status", "mps_available"]
