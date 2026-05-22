"""Branch coverage for ``visionforge.core.trainer.resolve_device``."""

from __future__ import annotations

from unittest.mock import patch

import torch

from visionforge.core.trainer import resolve_device
from visionforge.utils.config import DeviceConfig
from visionforge.utils.cuda import CUDAInfo, GPUDevice


def _info_no_cuda() -> CUDAInfo:
    return CUDAInfo(available=False)


def _info_single_gpu() -> CUDAInfo:
    return CUDAInfo(
        available=True,
        device_count=1,
        current_device=0,
        device_name="RTX 4090",
        cuda_version="12.4",
        devices=(
            GPUDevice(
                index=0,
                name="RTX 4090",
                total_memory_mb=24576,
                compute_capability="8.9",
            ),
        ),
    )


def _info_two_gpus() -> CUDAInfo:
    return CUDAInfo(
        available=True,
        device_count=2,
        current_device=0,
        device_name="RTX 4090",
        cuda_version="12.4",
        devices=(
            GPUDevice(
                index=0,
                name="RTX 4090",
                total_memory_mb=24576,
                compute_capability="8.9",
            ),
            GPUDevice(
                index=1,
                name="RTX 3090",
                total_memory_mb=24576,
                compute_capability="8.6",
            ),
        ),
    )


class TestResolveDeviceCPU:
    def test_cpu_kind_returns_cpu_regardless_of_cuda(self) -> None:
        with patch(
            "visionforge.core.trainer.check_cuda", return_value=_info_two_gpus()
        ):
            device, ids, label = resolve_device(DeviceConfig(kind="cpu"))
        assert device == torch.device("cpu")
        assert ids == []
        assert label == "cpu"

    def test_cuda_requested_but_unavailable_falls_back(self) -> None:
        with patch("visionforge.core.trainer.check_cuda", return_value=_info_no_cuda()):
            device, ids, label = resolve_device(DeviceConfig(kind="cuda"))
        assert device == torch.device("cpu")
        assert ids == []
        assert "fallback" in label.lower()


class TestResolveDeviceSingleCuda:
    def test_default_gpu_zero_when_no_ids_given(self) -> None:
        with patch(
            "visionforge.core.trainer.check_cuda", return_value=_info_two_gpus()
        ):
            device, ids, label = resolve_device(DeviceConfig(kind="cuda"))
        assert device == torch.device("cuda:0")
        assert ids == []
        assert "cuda:0" in label
        assert "RTX 4090" in label

    def test_explicit_valid_gpu_id_is_honoured(self) -> None:
        with patch(
            "visionforge.core.trainer.check_cuda", return_value=_info_two_gpus()
        ):
            device, ids, label = resolve_device(DeviceConfig(kind="cuda", gpu_ids=[1]))
        assert device == torch.device("cuda:1")
        assert ids == []
        assert "cuda:1" in label
        assert "RTX 3090" in label

    def test_invalid_gpu_id_falls_back_to_zero(self) -> None:
        """gpu_ids=[7] when only [0,1] exist must coerce to GPU 0 with a warning."""
        with patch(
            "visionforge.core.trainer.check_cuda", return_value=_info_two_gpus()
        ):
            device, ids, label = resolve_device(DeviceConfig(kind="cuda", gpu_ids=[7]))
        assert device == torch.device("cuda:0")
        assert ids == []
        assert "cuda:0" in label


class TestResolveDeviceMultiCuda:
    def test_default_uses_all_visible_gpus(self) -> None:
        with patch(
            "visionforge.core.trainer.check_cuda", return_value=_info_two_gpus()
        ):
            device, ids, label = resolve_device(DeviceConfig(kind="multi_cuda"))
        assert device == torch.device("cuda:0")
        assert ids == [0, 1]
        assert label.startswith("multi_cuda")
        assert "RTX 4090" in label and "RTX 3090" in label

    def test_explicit_ids_are_honoured(self) -> None:
        with patch(
            "visionforge.core.trainer.check_cuda", return_value=_info_two_gpus()
        ):
            device, ids, label = resolve_device(
                DeviceConfig(kind="multi_cuda", gpu_ids=[0, 1])
            )
        assert ids == [0, 1]
        assert label.startswith("multi_cuda")

    def test_falls_back_to_single_when_only_one_valid_id(self) -> None:
        """multi_cuda requested but only 1 valid GPU in available set → single CUDA fallback."""
        with patch(
            "visionforge.core.trainer.check_cuda", return_value=_info_single_gpu()
        ):
            # gpu_ids=[0,5]: 5 not present, so only 1 valid → fallback
            device, ids, label = resolve_device(
                DeviceConfig(kind="multi_cuda", gpu_ids=[0, 5])
            )
        assert device == torch.device("cuda:0")
        assert ids == []
        assert "cuda:0" in label

    def test_falls_back_when_no_valid_ids(self) -> None:
        """multi_cuda with all invalid ids → fallback to GPU 0 even if list ends up empty."""
        with patch(
            "visionforge.core.trainer.check_cuda", return_value=_info_single_gpu()
        ):
            device, ids, label = resolve_device(
                DeviceConfig(kind="multi_cuda", gpu_ids=[7, 8])
            )
        assert device == torch.device("cuda:0")
        assert ids == []
