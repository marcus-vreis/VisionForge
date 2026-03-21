from unittest.mock import MagicMock, patch

import pytest
from loguru import logger

from visionforge.utils.cuda import CUDAInfo, check_cuda, log_cuda_status


class TestCUDAInfo:
    def test_default_values(self) -> None:
        """Default CUDAInfo must represent 'no CUDA'."""
        info = CUDAInfo()
        assert info.available is False
        assert info.device_count == 0
        assert info.current_device is None
        assert info.device_name is None
        assert info.cuda_version is None

    def test_fully_populated(self) -> None:
        """CUDAInfo must accept all fields when explicitly provided."""
        info = CUDAInfo(
            available=True,
            device_count=2,
            current_device=0,
            device_name="NVIDIA RTX 4090",
            cuda_version="12.4",
        )
        assert info.available is True
        assert info.device_count == 2
        assert info.current_device == 0
        assert info.device_name == "NVIDIA RTX 4090"
        assert info.cuda_version == "12.4"

    def test_frozen(self) -> None:
        """CUDAInfo must be immutable (frozen dataclass)."""
        info = CUDAInfo()
        with pytest.raises(AttributeError):
            info.available = True  # type: ignore[misc]


class TestCheckCuda:
    def test_cuda_available_single_gpu(self) -> None:
        """When CUDA is available with one GPU, all fields must be populated."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.cuda.current_device.return_value = 0
        mock_torch.cuda.get_device_name.return_value = "NVIDIA RTX 4090"
        mock_torch.version.cuda = "12.4"

        with patch.dict("sys.modules", {"torch": mock_torch}):
            info = check_cuda()

        assert info.available is True
        assert info.device_count == 1
        assert info.current_device == 0
        assert info.device_name == "NVIDIA RTX 4090"
        assert info.cuda_version == "12.4"

    def test_cuda_not_available(self) -> None:
        """When CUDA is not available, check_cuda() must return a default CUDAInfo."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch.dict("sys.modules", {"torch": mock_torch}):
            info = check_cuda()

        assert info.available is False
        assert info.device_count == 0
        assert info.current_device is None
        assert info.device_name is None
        assert info.cuda_version is None

    def test_multiple_gpus(self) -> None:
        """Device count must be correctly reported for multi-GPU setups."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 4
        mock_torch.cuda.current_device.return_value = 0
        mock_torch.cuda.get_device_name.return_value = "NVIDIA A100"
        mock_torch.version.cuda = "12.2"

        with patch.dict("sys.modules", {"torch": mock_torch}):
            info = check_cuda()

        assert info.available is True
        assert info.device_count == 4

    def test_runtime_error_returns_unavailable(self) -> None:
        """If torch raises any exception, check_cuda() must return CUDAInfo(available=False)."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.side_effect = RuntimeError("driver error")

        with patch.dict("sys.modules", {"torch": mock_torch}):
            info = check_cuda()

        assert info.available is False
        assert info.device_count == 0

    def test_import_error_returns_unavailable(self) -> None:
        """If torch is missing, check_cuda() must return CUDAInfo(available=False)."""
        with patch.dict("sys.modules", {"torch": None}):
            info = check_cuda()

        assert info.available is False
        assert info.device_count == 0


class TestLogCudaStatus:
    @pytest.fixture(autouse=True)
    def _capture_logs(self):
        """Add a temporary loguru sink that captures log records."""
        self.log_messages: list[str] = []
        handler_id = logger.add(
            lambda msg: self.log_messages.append(str(msg)),
            format="{level} | {message}",
            level="DEBUG",
        )
        yield
        logger.remove(handler_id)

    @patch("visionforge.utils.cuda.check_cuda")
    def test_logs_info_when_cuda_available(self, mock_check: MagicMock) -> None:
        """An INFO-level message must be emitted when CUDA is present."""
        mock_check.return_value = CUDAInfo(
            available=True,
            device_count=1,
            current_device=0,
            device_name="NVIDIA RTX 4090",
            cuda_version="12.4",
        )

        log_cuda_status()

        combined = "".join(self.log_messages)
        assert "CUDA available" in combined

    @patch("visionforge.utils.cuda.check_cuda")
    def test_logs_warning_when_cuda_unavailable(self, mock_check: MagicMock) -> None:
        """A WARNING-level message must be emitted when CUDA is absent."""
        mock_check.return_value = CUDAInfo(available=False)

        log_cuda_status()

        combined = "".join(self.log_messages)
        assert "CUDA not available" in combined


class TestCudaExports:
    def test_all_exports(self) -> None:
        """__all__ must list exactly three public names."""
        from visionforge.utils import cuda

        assert set(cuda.__all__) == {"CUDAInfo", "check_cuda", "log_cuda_status"}

    def test_check_cuda_is_callable(self) -> None:
        """check_cuda must be callable."""
        assert callable(check_cuda)

    def test_log_cuda_status_is_callable(self) -> None:
        """log_cuda_status must be callable."""
        assert callable(log_cuda_status)
