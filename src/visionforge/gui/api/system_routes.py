"""System and model-default endpoints extracted under ADR-132."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import APIRouter

from visionforge.gui.api import system_service
from visionforge.gui.api.schemas import (
    DeviceInfoResponse,
    ModelDefaultsRequest,
    ModelDefaultsResponse,
    SystemInfo,
)
from visionforge.utils.cuda import CUDAInfo


def build_system_router(
    cuda_probe: Callable[[], CUDAInfo], worker_budget: Callable[[], int]
) -> APIRouter:
    """Build system endpoints under the parent router's language dependency."""
    router = APIRouter()

    @router.get("/system/info")
    async def get_system_info() -> SystemInfo:
        """Return host information and the memory-aware worker budget (ADR-098)."""
        return system_service.system_info(worker_budget())

    @router.get("/device/info")
    async def get_device_info() -> DeviceInfoResponse:
        """Probe the runtime and return the real list of selectable devices."""
        return system_service.device_info(cuda_probe())

    @router.post("/model/defaults")
    async def model_defaults(req: ModelDefaultsRequest) -> ModelDefaultsResponse:
        """Suggested settings for this architecture (ADR-099/100)."""
        return system_service.model_defaults(req)

    return router
