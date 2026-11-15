"""Apple's GPU (MPS) as an experimental device (#227, ADR-192).

CI has no Mac, so every probe is replaced: ``torch.backends.mps.is_available``
for the probe itself, and ``mps_available``/``check_cuda`` where the device is
resolved. ``torch.device("mps")`` can be built on any platform.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from visionforge.core.trainer import resolve_device
from visionforge.core.training_health import mps_not_deterministic
from visionforge.utils.config import DeviceConfig, TrainingConfig
from visionforge.utils.cuda import CUDAInfo, GPUDevice, mps_available

_TRAINER = "visionforge.core.trainer"


def _no_cuda() -> CUDAInfo:
    return CUDAInfo(available=False)


def _one_gpu() -> CUDAInfo:
    return CUDAInfo(
        available=True,
        device_count=1,
        current_device=0,
        device_name="RTX",
        cuda_version="12.8",
        devices=(GPUDevice(index=0, name="RTX", total_memory_mb=16000),),
    )


class TestProbe:
    def test_reports_what_torch_reports(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
        assert mps_available() is True
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        assert mps_available() is False

    def test_never_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom() -> bool:
            raise RuntimeError("no Metal")

        monkeypatch.setattr(torch.backends.mps, "is_available", boom)
        assert mps_available() is False

    def test_only_a_real_true_counts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A mocked torch answers a MagicMock, which is truthy.
        monkeypatch.setattr(torch.backends.mps, "is_available", MagicMock())
        assert mps_available() is False


class TestResolveDevice:
    def test_the_config_accepts_mps(self) -> None:
        assert DeviceConfig(kind="mps").kind == "mps"

    def test_mps_is_used_when_asked_for(self) -> None:
        with (
            patch(f"{_TRAINER}.check_cuda", return_value=_no_cuda()),
            patch(f"{_TRAINER}.mps_available", return_value=True),
        ):
            device, ids, label = resolve_device(DeviceConfig(kind="mps"))
        assert device == torch.device("mps")
        assert ids == []
        assert label == "mps (Apple GPU)"

    def test_mps_asked_for_but_absent_falls_back_to_cpu(self) -> None:
        with (
            patch(f"{_TRAINER}.check_cuda", return_value=_no_cuda()),
            patch(f"{_TRAINER}.mps_available", return_value=False),
        ):
            device, _, label = resolve_device(DeviceConfig(kind="mps"))
        assert device == torch.device("cpu")
        assert label == "cpu (fallback: MPS unavailable)"

    @pytest.mark.parametrize("kind", ["cuda", "multi_cuda"])
    def test_cuda_absent_falls_back_to_mps_with_the_reason(self, kind: str) -> None:
        with (
            patch(f"{_TRAINER}.check_cuda", return_value=_no_cuda()),
            patch(f"{_TRAINER}.mps_available", return_value=True),
        ):
            device, _, label = resolve_device(DeviceConfig(kind=kind))  # type: ignore[arg-type]
        assert device == torch.device("mps")
        assert label == "mps (fallback: CUDA unavailable)"

    def test_neither_accelerator_falls_back_to_cpu(self) -> None:
        with (
            patch(f"{_TRAINER}.check_cuda", return_value=_no_cuda()),
            patch(f"{_TRAINER}.mps_available", return_value=False),
        ):
            device, _, label = resolve_device(DeviceConfig(kind="cuda"))
        assert device == torch.device("cpu")
        assert label == "cpu (fallback: CUDA unavailable)"

    def test_cuda_wins_when_both_exist(self) -> None:
        with (
            patch(f"{_TRAINER}.check_cuda", return_value=_one_gpu()),
            patch(f"{_TRAINER}.mps_available", return_value=True),
        ):
            device, _, _ = resolve_device(DeviceConfig(kind="cuda"))
        assert device == torch.device("cuda:0")


class TestDeterminismWarning:
    def test_deterministic_on_mps_warns(self) -> None:
        warning = mps_not_deterministic("mps", deterministic=True)

        assert warning is not None
        assert warning.code == "mps_not_deterministic"

    @pytest.mark.parametrize(
        ("device_type", "deterministic"),
        [("mps", False), ("cpu", True), ("cuda", True)],
    )
    def test_nothing_to_say_otherwise(
        self, device_type: str, deterministic: bool
    ) -> None:
        assert mps_not_deterministic(device_type, deterministic) is None

    @pytest.mark.parametrize(("device_type", "expected"), [("mps", 1), ("cpu", 0)])
    def test_the_classification_warnings_carry_it(
        self, device_type: str, expected: int
    ) -> None:
        """``Trainer._health_warnings`` is what the run.json ``warnings`` is."""
        from visionforge.core.trainer import Trainer

        trainer = Trainer.__new__(Trainer)
        trainer._config = SimpleNamespace(  # type: ignore[assignment]
            task="binary",
            model=SimpleNamespace(num_classes=2),
            training=TrainingConfig(),
        )
        trainer._device = torch.device(device_type)
        trainer._last_val_preds = []

        codes = [w["code"] for w in trainer._health_warnings([])]

        assert codes.count("mps_not_deterministic") == expected


def _detection_trainer(kind: str) -> Any:
    """A DetectionTrainer shell holding only the config the device helpers read."""
    from visionforge.core.detection_trainer import DetectionTrainer

    trainer = DetectionTrainer.__new__(DetectionTrainer)
    trainer._config = SimpleNamespace(  # type: ignore[assignment]
        device=DeviceConfig(kind=kind),  # type: ignore[arg-type]
        training=SimpleNamespace(deterministic=True),
    )
    return trainer


@pytest.fixture
def mac(monkeypatch: pytest.MonkeyPatch) -> None:
    import visionforge.core.detection_trainer as mod

    monkeypatch.setattr(mod.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(mod, "mps_available", lambda: True)


@pytest.fixture
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    import visionforge.core.detection_trainer as mod

    monkeypatch.setattr(mod.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(mod, "mps_available", lambda: False)


class TestDetectionOnAMac:
    @pytest.mark.usefixtures("mac")
    @pytest.mark.parametrize("kind", ["mps", "cuda", "multi_cuda"])
    def test_ultralytics_gets_mps(self, kind: str) -> None:
        assert _detection_trainer(kind)._device_arg() == "mps"

    @pytest.mark.usefixtures("mac")
    def test_torchvision_gets_mps(self) -> None:
        assert _detection_trainer("mps")._resolve_torch_device() == torch.device("mps")

    @pytest.mark.usefixtures("mac")
    def test_the_label_names_mps(self) -> None:
        assert _detection_trainer("cuda")._resolve_device_label() == "mps"

    @pytest.mark.usefixtures("mac")
    def test_cpu_stays_cpu(self) -> None:
        assert _detection_trainer("cpu")._device_arg() == "cpu"

    @pytest.mark.usefixtures("mac")
    def test_deterministic_run_is_warned(self) -> None:
        warning = _detection_trainer("mps")._mps_warning()

        assert warning is not None
        assert warning.code == "mps_not_deterministic"


class TestDetectionWithoutAnyAccelerator:
    @pytest.mark.usefixtures("cpu_only")
    @pytest.mark.parametrize("kind", ["mps", "cuda", "multi_cuda"])
    def test_everything_falls_back_to_cpu(self, kind: str) -> None:
        trainer = _detection_trainer(kind)

        assert trainer._device_arg() == "cpu"
        assert trainer._resolve_torch_device() == torch.device("cpu")
        assert trainer._resolve_device_label() == "cpu"
        assert trainer._mps_warning() is None


class TestDetectionWithCuda:
    def test_cuda_is_kept(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import visionforge.core.detection_trainer as mod

        monkeypatch.setattr(mod.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(mod, "mps_available", lambda: True)
        trainer = _detection_trainer("multi_cuda")

        assert trainer._device_arg() == 0
        assert trainer._resolve_device_label() == "multi_cuda:all"
