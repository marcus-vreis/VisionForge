from __future__ import annotations

import platform

from visionforge.utils.environment import capture_environment


class TestCaptureEnvironment:
    def test_returns_expected_keys(self) -> None:
        env = capture_environment()
        for key in (
            "python",
            "platform",
            "torch",
            "torchvision",
            "numpy",
            "visionforge",
            "cuda",
            "cudnn",
            "gpu",
        ):
            assert key in env

    def test_all_values_are_strings(self) -> None:
        env = capture_environment()
        assert all(isinstance(v, str) for v in env.values())

    def test_python_version_matches_runtime(self) -> None:
        env = capture_environment()
        assert env["python"] == platform.python_version()

    def test_torch_version_present(self) -> None:
        # torch is a hard dependency of the runtime — its version must resolve.
        env = capture_environment()
        assert env["torch"] != "unknown"
        assert env["torch"][0].isdigit()

    def test_unresolved_package_falls_back_to_unknown(self) -> None:
        # A package that is not installed must not raise — it reports "unknown".
        from visionforge.utils.environment import _safe_version

        assert _safe_version("definitely-not-a-real-package-xyz") == "unknown"

    def test_torch_runtime_probe_reports_cuda_build_or_none(self) -> None:
        # On a CPU-only build cuda/gpu report "none"; on a CUDA build the CUDA
        # version resolves. Either way the probe never reports a probe failure
        # when torch imports cleanly.
        env = capture_environment()
        assert env["cuda"] != "unknown"
        assert env["gpu"] != "unknown"


def test_visionforge_version_is_the_installed_distribution() -> None:
    # The distribution is "visionforge-studio"; looking up the import name
    # reported "unknown" in every run.json.
    import visionforge

    env = capture_environment()
    assert env["visionforge"] == visionforge.__version__
    assert env["visionforge"] != "unknown"


def test_a_checkout_records_the_version_the_package_reports(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # In a checkout __version__ comes from pyproject.toml while the installed
    # metadata lags until a reinstall; run.json must record what the package says
    # (the 0.16.0 bump caught the two disagreeing).
    import visionforge

    monkeypatch.setattr(visionforge, "__version__", "9.9.9")
    assert capture_environment()["visionforge"] == "9.9.9"
