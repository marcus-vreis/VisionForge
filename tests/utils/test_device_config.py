"""Validator coverage for ``visionforge.utils.config.DeviceConfig``."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from visionforge.utils.config import DeviceConfig


class TestDeviceConfigDefaults:
    def test_default_is_single_cuda(self) -> None:
        """Without overrides, DeviceConfig requests CUDA on GPU 0."""
        cfg = DeviceConfig()
        assert cfg.kind == "cuda"
        assert cfg.gpu_ids is None

    def test_cpu_kind_accepts_no_ids(self) -> None:
        cfg = DeviceConfig(kind="cpu")
        assert cfg.kind == "cpu"
        assert cfg.gpu_ids is None


class TestGpuIdsValidator:
    def test_negative_id_rejected(self) -> None:
        with pytest.raises(ValidationError, match=r"gpu_ids must all be >= 0"):
            DeviceConfig(kind="cuda", gpu_ids=[-1])

    def test_negative_id_in_list_rejected(self) -> None:
        with pytest.raises(ValidationError, match=r"gpu_ids must all be >= 0"):
            DeviceConfig(kind="multi_cuda", gpu_ids=[0, -2])

    def test_duplicate_ids_rejected(self) -> None:
        with pytest.raises(ValidationError, match=r"gpu_ids must be unique"):
            DeviceConfig(kind="multi_cuda", gpu_ids=[0, 1, 0])

    def test_empty_list_is_accepted(self) -> None:
        """An empty list is degenerate but not invalid at the field level."""
        cfg = DeviceConfig(kind="cuda", gpu_ids=[])
        assert cfg.gpu_ids == []

    def test_valid_ids_accepted(self) -> None:
        cfg = DeviceConfig(kind="multi_cuda", gpu_ids=[0, 1, 2, 3])
        assert cfg.gpu_ids == [0, 1, 2, 3]


class TestMultiCudaValidator:
    def test_multi_cuda_with_single_id_rejected(self) -> None:
        with pytest.raises(ValidationError, match=r"multi_cuda requires at least 2"):
            DeviceConfig(kind="multi_cuda", gpu_ids=[0])

    def test_multi_cuda_with_two_ids_accepted(self) -> None:
        cfg = DeviceConfig(kind="multi_cuda", gpu_ids=[0, 1])
        assert cfg.kind == "multi_cuda"
        assert cfg.gpu_ids == [0, 1]

    def test_multi_cuda_without_explicit_ids_accepted(self) -> None:
        """multi_cuda with gpu_ids=None means 'use all visible'; runtime handles count."""
        cfg = DeviceConfig(kind="multi_cuda")
        assert cfg.gpu_ids is None
