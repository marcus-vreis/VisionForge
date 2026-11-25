"""System and model-default endpoints extracted under ADR-132."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from fastapi import APIRouter

from visionforge.gui.api import system_service
from visionforge.gui.api.http_responses import TAG_EXPERIMENTS, TAG_SYSTEM
from visionforge.gui.api.schemas import (
    CustomModelsResponse,
    DeviceInfoResponse,
    LearningRatePlanRequest,
    LearningRatePlanResponse,
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

    @router.get("/system/info", tags=[TAG_SYSTEM])
    async def get_system_info() -> SystemInfo:
        """Return host information and the memory-aware worker budget (ADR-098)."""
        return system_service.system_info(worker_budget())

    @router.get("/device/info", tags=[TAG_SYSTEM])
    async def get_device_info() -> DeviceInfoResponse:
        """Probe the runtime and return the real list of selectable devices."""
        return system_service.device_info(cuda_probe())

    @router.post("/model/defaults", tags=[TAG_EXPERIMENTS])
    async def model_defaults(req: ModelDefaultsRequest) -> ModelDefaultsResponse:
        """Suggested settings for this architecture (ADR-099/100)."""
        # Sampling ``base_dir`` walks the folder and opens images: a large tree
        # must not hold the event loop every other request waits on.
        return await asyncio.to_thread(system_service.model_defaults, req)

    @router.post("/model/learning-rate-plan", tags=[TAG_EXPERIMENTS])
    async def learning_rate_plan(
        req: LearningRatePlanRequest,
    ) -> LearningRatePlanResponse:
        """The rate each trial of this job will train at (ADR-158)."""
        return system_service.learning_rate_plan(req)

    # Plain ``def``: the scan imports the user's files, and FastAPI runs a sync
    # handler in a worker thread instead of on the event loop.
    @router.get("/model/custom", tags=[TAG_EXPERIMENTS])
    def custom_models() -> CustomModelsResponse:
        """The models of ``user_models/`` that ``model.custom_model`` can name."""
        return system_service.custom_models()

    return router
