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
