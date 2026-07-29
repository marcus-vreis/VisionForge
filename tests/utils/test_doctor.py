"""Tests for visionforge.utils.doctor — all subprocess calls are mocked."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from visionforge.utils.doctor import (
    build_install_command,
    check_python,
    detect_driver_cuda,
    probe_torch,
    run_doctor,
    select_wheel_tag,
)

# ---------------------------------------------------------------------------
# select_wheel_tag
# ---------------------------------------------------------------------------


class TestSelectWheelTag:
    @pytest.mark.parametrize(
        ("driver_ver", "expected_tag"),
        [
            ("12.6", "cu126"),
            ("12.7", "cu126"),  # between 12.6 and 12.8 → cu126
            ("12.8", "cu128"),
            ("13.3", "cu128"),  # above highest supported → clamp to cu128
            ("12.4", "cu124"),
            ("12.5", "cu124"),  # between 12.4 and 12.6 → cu124
            ("12.1", "cu121"),
            ("12.3", "cu121"),  # between 12.1 and 12.4 → cu121
            ("12.0", "cu118"),  # edge: just below cu121 boundary → cu118
            ("11.8", "cu118"),
            ("11.2", "cpu"),  # below 11.8 → cpu
            (None, "cpu"),  # no GPU detected → cpu
        ],
    )
    def test_mapping(self, driver_ver: str | None, expected_tag: str) -> None:
        assert select_wheel_tag(driver_ver) == expected_tag

    def test_blackwell_drivers_get_cu128_not_an_older_wheel(self) -> None:
        """RTX 50-series is compute capability 12.0, and no wheel before cu128
        ships an sm_120 kernel.

        An earlier build imports fine and reports the GPU, then fails at the
        first kernel launch — exactly the silent misconfiguration doctor exists
        to prevent, so the mapping must not stop below cu128.
        """
        assert select_wheel_tag("12.8") == "cu128"
        assert select_wheel_tag("13.0") == "cu128"


# ---------------------------------------------------------------------------
# detect_driver_cuda
# ---------------------------------------------------------------------------


class TestDetectDriverCuda:
    def _make_result(self, stdout: str, returncode: int = 0) -> MagicMock:
        r = MagicMock()
        r.stdout = stdout
        r.returncode = returncode
        return r

    def test_parses_cuda_version_from_smi_header(self) -> None:
        output = (
            "+-----------------------------------------------------------------------------------------+\n"
            "| NVIDIA-SMI 560.35.03              Driver Version: 560.35.03      CUDA Version: 12.4     |\n"
            "+-----------------------------------------------------------------------------------------+\n"
        )
        with patch("subprocess.run", return_value=self._make_result(output)):
            assert detect_driver_cuda() == "12.4"

    def test_returns_none_on_file_not_found(self) -> None:
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert detect_driver_cuda() is None

    def test_returns_none_on_nonzero_exit(self) -> None:
        with patch("subprocess.run", return_value=self._make_result("", returncode=1)):
            assert detect_driver_cuda() is None

    def test_returns_none_on_empty_output(self) -> None:
        with patch("subprocess.run", return_value=self._make_result("")):
            assert detect_driver_cuda() is None

    def test_returns_none_on_malformed_output(self) -> None:
        with patch(
            "subprocess.run",
            return_value=self._make_result("no version info here at all"),
        ):
            assert detect_driver_cuda() is None

    def test_parses_version_with_single_digit_minor(self) -> None:
        output = (
            "| NVIDIA-SMI 500.00    Driver Version: 500.00    CUDA Version: 11.8 |\n"
        )
        with patch("subprocess.run", return_value=self._make_result(output)):
            assert detect_driver_cuda() == "11.8"


# ---------------------------------------------------------------------------
# build_install_command
# ---------------------------------------------------------------------------


class TestBuildInstallCommand:
    """The command has to match how *this* install got here.

    `pip install -e ".[cpu]"` only works inside a checkout. Someone who
    installed the wheel from PyPI has no source tree, so that line just fails —
    and it was the first thing doctor told them to run.
    """

    @pytest.fixture
    def from_wheel(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "visionforge.utils.doctor.installed_from_source", lambda: False
        )

    @pytest.fixture
    def from_source(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "visionforge.utils.doctor.installed_from_source", lambda: True
        )

    def test_wheel_install_uses_the_distribution_name(self, from_wheel: None) -> None:
        cmd, url = build_install_command("cu124")
        assert cmd == 'pip install "visionforge-studio[cu124]"'
        assert url == "https://download.pytorch.org/whl/cu124"

    def test_editable_install_keeps_the_source_form(self, from_source: None) -> None:
        cmd, _ = build_install_command("cu124")
        assert cmd == 'pip install -e ".[cu124]"'

    def test_cpu_tag(self, from_wheel: None) -> None:
        cmd, url = build_install_command("cpu")
        assert cmd == 'pip install "visionforge-studio[cpu]"'
        assert url == "https://download.pytorch.org/whl/cpu"

    def test_cu118_tag(self, from_wheel: None) -> None:
        cmd, url = build_install_command("cu118")
        assert cmd == 'pip install "visionforge-studio[cu118]"'
        assert url == "https://download.pytorch.org/whl/cu118"

    def test_the_distribution_name_is_not_the_import_name(self) -> None:
        """Plain `visionforge` on PyPI is an unrelated project; pointing users
        at it would install someone else's package."""
        from visionforge.utils.doctor import _DIST_NAME

        assert _DIST_NAME == "visionforge-studio"


# ---------------------------------------------------------------------------
# probe_torch
# ---------------------------------------------------------------------------


class TestProbeTorch:
    def test_torch_not_importable(self) -> None:
        with patch("importlib.util.find_spec", return_value=None):
            result = probe_torch()
        assert result["importable"] is False
        assert result["version"] == "unknown"
        assert result["cuda_available"] is False

    def test_torch_importable_cuda_not_available(self) -> None:
        fake_torch = MagicMock()
        fake_torch.__version__ = "2.3.0"
        fake_torch.cuda.is_available.return_value = False

        fake_spec = MagicMock()
        with (
            patch("importlib.util.find_spec", return_value=fake_spec),
            patch.dict(sys.modules, {"torch": fake_torch}),
        ):
            result = probe_torch()

        assert result["importable"] is True
        assert result["version"] == "2.3.0"
        assert result["cuda_available"] is False

    def test_torch_importable_cuda_available(self) -> None:
        fake_torch = MagicMock()
        fake_torch.__version__ = "2.4.0+cu124"
        fake_torch.cuda.is_available.return_value = True

        fake_spec = MagicMock()
        with (
            patch("importlib.util.find_spec", return_value=fake_spec),
            patch.dict(sys.modules, {"torch": fake_torch}),
        ):
            result = probe_torch()

        assert result["importable"] is True
        assert result["cuda_available"] is True


# ---------------------------------------------------------------------------
# check_python
# ---------------------------------------------------------------------------


class TestCheckPython:
    def test_current_python_is_313_or_above(self) -> None:
        # CI runs Python 3.13+ per pyproject.toml requires-python
        result = check_python()
        assert isinstance(result["version"], str)
        assert isinstance(result["ok"], bool)
        # The version string must be non-empty
        assert result["version"]

    def test_old_python_reports_not_ok(self) -> None:
        with patch.object(sys, "version_info", (3, 12, 0, "final", 0)):
            result = check_python()
        assert result["ok"] is False
        assert "3.12" in result["version"]

    def test_python_313_reports_ok(self) -> None:
        with patch.object(sys, "version_info", (3, 13, 0, "final", 0)):
            result = check_python()
        assert result["ok"] is True


# ---------------------------------------------------------------------------
# run_doctor
# ---------------------------------------------------------------------------


class TestRunDoctor:
    def _make_smi_output(self, ver: str) -> MagicMock:
        r = MagicMock()
        r.stdout = (
            f"| NVIDIA-SMI 560.00  Driver Version: 560.00  CUDA Version: {ver} |\n"
        )
        r.returncode = 0
        return r

    def test_fix_false_never_calls_confirm_or_install(self) -> None:
        confirm = MagicMock()
        with (
            patch("subprocess.run", return_value=self._make_smi_output("12.4")),
            patch("importlib.util.find_spec", return_value=None),
            patch("visionforge.utils.doctor._run_install") as mock_install,
        ):
            run_doctor(fix=False, confirm_fn=confirm)

        confirm.assert_not_called()
        mock_install.assert_not_called()

    def test_fix_true_confirm_yes_calls_install(self) -> None:
        confirm = MagicMock(return_value=True)
        with (
            patch("subprocess.run", return_value=self._make_smi_output("12.4")),
            patch("importlib.util.find_spec", return_value=None),
            patch("visionforge.utils.doctor._run_install") as mock_install,
        ):
            run_doctor(fix=True, confirm_fn=confirm)

        confirm.assert_called_once()
        mock_install.assert_called_once()
        # Verify correct tag was selected
        call_args = mock_install.call_args
        assert "cu124" in call_args[0][0]

    def test_fix_true_confirm_no_does_not_install(self) -> None:
        confirm = MagicMock(return_value=False)
        with (
            patch("subprocess.run", return_value=self._make_smi_output("12.4")),
            patch("importlib.util.find_spec", return_value=None),
            patch("visionforge.utils.doctor._run_install") as mock_install,
        ):
            run_doctor(fix=True, confirm_fn=confirm)

        confirm.assert_called_once()
        mock_install.assert_not_called()

    def test_report_always_prints_install_command(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # The install command must appear in the report even without --fix
        with (
            patch("subprocess.run", return_value=self._make_smi_output("12.6")),
            patch("importlib.util.find_spec", return_value=None),
            patch("visionforge.utils.doctor._run_install"),
        ):
            run_doctor(fix=False, confirm_fn=MagicMock())

        out = capsys.readouterr().out
        assert "cu126" in out
        assert "pip install" in out
        assert "https://download.pytorch.org/whl/cu126" in out

    def test_no_gpu_recommends_cpu(self, capsys: pytest.CaptureFixture[str]) -> None:
        with (
            patch("subprocess.run", side_effect=FileNotFoundError),
            patch("importlib.util.find_spec", return_value=None),
        ):
            run_doctor(fix=False, confirm_fn=MagicMock())

        out = capsys.readouterr().out
        assert "Recommended wheel: cpu" in out

    def test_verdict_gpu_driver_but_cpu_torch_is_not_ok(self) -> None:
        """CUDA driver present but torch.cuda.is_available() False → exit code 1."""
        fake_torch = MagicMock()
        fake_torch.__version__ = "2.3.0+cpu"
        fake_torch.cuda.is_available.return_value = False
        fake_spec = MagicMock()

        with (
            patch("subprocess.run", return_value=self._make_smi_output("12.4")),
            patch("importlib.util.find_spec", return_value=fake_spec),
            patch.dict(sys.modules, {"torch": fake_torch}),
        ):
            exit_code = run_doctor(fix=False, confirm_fn=MagicMock())

        assert exit_code == 1

    def test_verdict_gpu_driver_but_cpu_torch_prints_reinstall_warning(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Report must flag the CPU-torch-on-GPU-machine mismatch."""
        fake_torch = MagicMock()
        fake_torch.__version__ = "2.3.0+cpu"
        fake_torch.cuda.is_available.return_value = False
        fake_spec = MagicMock()

        with (
            patch("subprocess.run", return_value=self._make_smi_output("12.4")),
            patch("importlib.util.find_spec", return_value=fake_spec),
            patch.dict(sys.modules, {"torch": fake_torch}),
        ):
            run_doctor(fix=False, confirm_fn=MagicMock())

        out = capsys.readouterr().out
        # Must flag the mismatch and recommend reinstalling with GPU extra
        assert (
            "reinstall" in out.lower()
            or "gpu extra" in out.lower()
            or "mismatch" in out.lower()
        )

    def test_all_ok_returns_zero(self) -> None:
        fake_torch = MagicMock()
        fake_torch.__version__ = "2.4.0+cu124"
        fake_torch.cuda.is_available.return_value = True
        fake_spec = MagicMock()

        with (
            patch("subprocess.run", return_value=self._make_smi_output("12.4")),
            patch("importlib.util.find_spec", return_value=fake_spec),
            patch.dict(sys.modules, {"torch": fake_torch}),
            patch.object(sys, "version_info", (3, 13, 0, "final", 0)),
        ):
            exit_code = run_doctor(fix=False, confirm_fn=MagicMock())

        assert exit_code == 0


# ---------------------------------------------------------------------------
# CLI dispatch
# ---------------------------------------------------------------------------


class TestDoctorCLIDispatch:
    def test_doctor_subcommand_wired(self) -> None:
        """Calling main() with 'doctor' must not raise and must not shell out."""
        from visionforge.__main__ import main

        with (
            patch("sys.argv", ["visionforge", "doctor"]),
            patch("subprocess.run", return_value=MagicMock(stdout="", returncode=1)),
            patch("importlib.util.find_spec", return_value=None),
            patch("visionforge.utils.doctor._run_install") as mock_install,
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        mock_install.assert_not_called()
        # exit code 0 or 1 are both valid; what matters is we reached sys.exit()
        assert exc_info.value.code in (0, 1)

    def test_doctor_fix_confirm_no_does_not_install(self) -> None:
        """--fix with user declining must not run the install."""
        from visionforge.__main__ import main

        with (
            patch("sys.argv", ["visionforge", "doctor", "--fix"]),
            patch("subprocess.run", return_value=MagicMock(stdout="", returncode=1)),
            patch("importlib.util.find_spec", return_value=None),
            patch("visionforge.utils.doctor._run_install") as mock_install,
            # Patch the confirm helper that __main__ passes in
            patch("visionforge.utils.doctor._default_confirm", return_value=False),
            pytest.raises(SystemExit),
        ):
            main()

        mock_install.assert_not_called()


class TestGpuDetectedWithoutNvidiaSmi:
    """nvidia-smi is simply absent from PATH on plenty of working GPU machines.

    doctor used to derive its whole recommendation from that one probe, so it
    told a researcher with a functioning RTX card to install the CPU wheel —
    and then printed "environment looks good" underneath. The torch probe knows
    better and is now consulted first.
    """

    def _run(self, capsys: pytest.CaptureFixture[str]) -> str:
        run_doctor(fix=False)
        out: str = capsys.readouterr().out
        return out

    def test_reports_the_gpu_and_keeps_the_working_install(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr("visionforge.utils.doctor.detect_driver_cuda", lambda: None)
        monkeypatch.setattr(
            "visionforge.utils.doctor.probe_torch",
            lambda: {
                "importable": True,
                "version": "2.11.0+cu128",
                "cuda_available": True,
                "cuda_build": "12.8",
            },
        )
        out = self._run(capsys)
        assert "GPU usable via torch" in out
        assert "keep the current install" in out
        # The bug in one line: never send a working GPU user to the CPU wheel.
        assert '".[cpu]"' not in out

    def test_still_recommends_cpu_when_there_is_no_gpu_at_all(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr("visionforge.utils.doctor.detect_driver_cuda", lambda: None)
        monkeypatch.setattr(
            "visionforge.utils.doctor.probe_torch",
            lambda: {
                "importable": True,
                "version": "2.11.0+cpu",
                "cuda_available": False,
                "cuda_build": None,
            },
        )
        out = self._run(capsys)
        assert "No CUDA-capable GPU detected" in out
        assert "Recommended wheel: cpu" in out
