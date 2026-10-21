"""System probe payloads and model suggestions, independent of HTTP routing."""

from __future__ import annotations

import os
import platform
from dataclasses import asdict
from pathlib import Path

from visionforge import __version__
from visionforge.gui.api.schemas import (
    CollapseEvidenceResponse,
    DeviceInfoResponse,
    GPUInfo,
    ModelDefaultsRequest,
    ModelDefaultsResponse,
    SystemInfo,
)
from visionforge.utils.cuda import CUDAInfo
from visionforge.utils.messages import tr


def system_info(suggested_workers: int) -> SystemInfo:
    """Describe the host using the worker budget supplied for this request."""
    return SystemInfo(
        cpu_count=os.cpu_count() or 1,
        suggested_workers=suggested_workers,
        platform=platform.system(),
        version=__version__,
    )


def device_info(info: CUDAInfo) -> DeviceInfoResponse:
    """Describe selectable devices from a fresh CUDA probe."""
    return DeviceInfoResponse(
        cuda_available=info.available,
        cuda_version=info.cuda_version,
        cpu_name=platform.processor() or platform.machine() or "CPU",
        gpus=[
            GPUInfo(
                index=d.index,
                name=d.name,
                total_memory_mb=d.total_memory_mb,
                compute_capability=d.compute_capability,
            )
            for d in info.devices
        ],
    )


def model_defaults(req: ModelDefaultsRequest) -> ModelDefaultsResponse:
    """Suggest model settings and explain measured failures in the current language."""
    # Importing core loads its training exports; defer them until a suggestion
    # is requested so importing this service or building its router stays idle.
    from visionforge.core.image_size import median_image_side, suggested_image_size
    from visionforge.core.learning_rate import (
        collapse_evidence,
        is_collapse_prone,
        suggested_learning_rate,
        suggested_optimizer,
    )
    from visionforge.gui.api.model_notes import collapse_note

    arch = req.architecture
    optimizer = suggested_optimizer(arch)
    rate = suggested_learning_rate(arch, optimizer)

    median: int | None = None
    size: int | None = None
    if req.base_dir:
        root = Path(req.base_dir)
        median = median_image_side(root)
        size = suggested_image_size(root, arch, pretrained=req.pretrained)

    note: str | None = None
    prone = is_collapse_prone(arch, "adam", 1e-3)
    evidence = collapse_evidence(arch) if prone else None
    if prone:
        note = collapse_note(arch, optimizer, rate, evidence)
    elif median is not None and size is not None and median < size:
        note = tr("model.small_images", median=median)

    return ModelDefaultsResponse(
        architecture=arch,
        optimizer=optimizer,
        learning_rate=rate,
        image_size=size,
        dataset_median_side=median,
        collapse_prone=prone,
        collapse_evidence=(
            CollapseEvidenceResponse.model_validate(asdict(evidence))
            if evidence is not None
            else None
        ),
        note=note,
    )
