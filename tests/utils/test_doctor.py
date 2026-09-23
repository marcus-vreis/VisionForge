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
    torch_swap_steps,
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

    def test_parses_umd_version_from_driver_6xx_header(self) -> None:
        """Driver 6xx renamed the field, which used to read as "no GPU at all"."""
        output = (
            "+-----------------------------------------------------------------------------------------+\n"
            "| NVIDIA-SMI 610.74                 KMD Version: 610.74        CUDA UMD Version: 13.3     |\n"
            "+-----------------------------------------------------------------------------------------+\n"
        )
        with patch("subprocess.run", return_value=self._make_result(output)):
            assert detect_driver_cuda() == "13.3"

    def test_driver_6xx_still_selects_a_cuda_wheel(self) -> None:
        """The bug's real cost: a CUDA machine being told to install cpu torch."""
        assert select_wheel_tag("13.3") == "cu128"

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
    """The fix has to replace the torch that is already here (ADR-107).

    Since ADR-106 the first install always brings a torch — `ultralytics` pulls
    one from PyPI, and on Windows PyPI only has CPU builds. The old advice,
    `pip install "visionforge-studio[cu128]" --index-url …/cu128`, then printed
    "Requirement already satisfied: torch>=2.3" and changed nothing, so a GPU
    user who answered `y` to `--fix` kept training on the CPU.
    """

    @pytest.mark.parametrize("tag", ["cpu", "cu118", "cu124", "cu128"])
    def test_removes_then_installs_from_that_index(self, tag: str) -> None:
        commands, url = build_install_command(tag)

        assert url == f"https://download.pytorch.org/whl/{tag}"
        assert commands == [
            "pip uninstall -y torch torchvision",
            f"pip install torch torchvision --index-url {url}",
        ]

    def test_never_reinstalls_the_package_itself(self) -> None:
        """It is already installed — that is how doctor is running — and it is
        not on the PyTorch index, so naming it there could only fail."""
        commands, _ = build_install_command("cu128")

        assert not any("visionforge" in line for line in commands)

    def test_one_command_per_line(self) -> None:
        """Windows PowerShell 5.1 has no `&&`, so a chained line would fail
        exactly where most of these users are."""
        commands, _ = build_install_command("cu128")

        assert not any("&&" in line for line in commands)


class TestTorchSwapSteps:
    """What `--fix` actually executes, and into which environment."""

    def test_uses_this_interpreters_pip(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Not whatever `pip` is first on PATH: with the venv not activated that
        is another Python, and the CUDA build lands where nothing imports it."""
        monkeypatch.setattr("importlib.util.find_spec", lambda _name: object())

        uninstall, install = torch_swap_steps("cu128")

        assert uninstall[:3] == [sys.executable, "-m", "pip"]
        assert uninstall[3:] == ["uninstall", "-y", "torch", "torchvision"]
        assert install[:3] == [sys.executable, "-m", "pip"]
        assert install[-2:] == ["--index-url", "https://download.pytorch.org/whl/cu128"]

    def test_falls_back_to_uv_where_there_is_no_pip(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`uv venv` makes environments without pip; the dev checkout is one."""
        monkeypatch.setattr("importlib.util.find_spec", lambda _name: None)
        monkeypatch.setattr("shutil.which", lambda _name: "C:/bin/uv.exe")

        uninstall, install = torch_swap_steps("cu128")

        assert uninstall[:3] == ["C:/bin/uv.exe", "pip", "uninstall"]
        assert uninstall[-2:] == ["--python", sys.executable]
        assert install[-2:] == ["--python", sys.executable]
        assert "--index-url" in install

    def test_uninstall_runs_first_and_the_install_decides_the_exit_code(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from visionforge.utils import doctor

        calls: list[list[str]] = []

        def fake_run(argv: list[str], check: bool) -> MagicMock:
            calls.append(argv)
            result = MagicMock()
            # A torch that is not there to remove is fine; the install failing
            # is what the caller has to hear about.
            result.returncode = 1 if "uninstall" in argv else 0
            return result

        monkeypatch.setattr("importlib.util.find_spec", lambda _name: object())
        monkeypatch.setattr(doctor.subprocess, "run", fake_run)

        code = doctor._run_install("cu128")

        assert code == 0
        assert "uninstall" in calls[0]
        assert "install" in calls[1] and "uninstall" not in calls[1]

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

    @staticmethod
    def _cuda_torch(version: str, build: str) -> MagicMock:
        fake = MagicMock()
        fake.__version__ = version
        fake.cuda.is_available.return_value = True
        fake.version.cuda = build
        return fake

    def test_a_right_install_is_never_told_to_uninstall(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The fix now starts with `pip uninstall` (ADR-107). Printing that to
        someone whose CUDA build is already the one this driver calls for would
        be advice to break a working setup."""
        with (
            patch("subprocess.run", return_value=self._make_smi_output("13.3")),
            patch("importlib.util.find_spec", return_value=MagicMock()),
            patch.dict(
                sys.modules, {"torch": self._cuda_torch("2.11.0+cu128", "12.8")}
            ),
        ):
            run_doctor(fix=False, confirm_fn=MagicMock())

        out = capsys.readouterr().out
        assert "keep the current install" in out
        assert "uninstall" not in out

    def test_an_older_cuda_build_still_gets_the_commands(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """ "Runs on GPU" is not enough: an older build on an RTX 50 imports,
        reports the GPU, and fails at the first kernel launch."""
        with (
            patch("subprocess.run", return_value=self._make_smi_output("13.3")),
            patch("importlib.util.find_spec", return_value=MagicMock()),
            patch.dict(sys.modules, {"torch": self._cuda_torch("2.4.0+cu124", "12.4")}),
        ):
            run_doctor(fix=False, confirm_fn=MagicMock())

        out = capsys.readouterr().out
        assert "Recommended wheel: cu128" in out
        assert "--index-url https://download.pytorch.org/whl/cu128" in out

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
