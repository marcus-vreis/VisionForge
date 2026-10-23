from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path

import pytest

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


def test_records_extended_dependency_versions() -> None:
    env = capture_environment()
    for name in (
        "scipy",
        "scikit-learn",
        "ultralytics",
        "timm",
        "optuna",
        "pillow",
        "pydantic",
    ):
        assert name in env
        assert isinstance(env[name], str)


def _source_checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from visionforge.utils import environment

    root = tmp_path / "checkout"
    source = root / "src" / "visionforge" / "utils" / "environment.py"
    source.parent.mkdir(parents=True)
    source.write_text("", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        '[project]\nname = "visionforge-studio"\n', encoding="utf-8"
    )
    # Commit hooks export repository routing even when Git runs with another cwd.
    git_env = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    for args in (
        ["init", "--quiet"],
        ["add", "."],
        [
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.org",
            "commit",
            "-qm",
            "initial",
        ],
    ):
        subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            env=git_env,
        )
    monkeypatch.setattr(environment, "__file__", str(source))
    return root


def test_source_fixture_isolates_inherited_git_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A commit hook must not redirect the fixture into its caller's repository."""
    clean_env = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    foreign = tmp_path / "foreign.git"
    subprocess.run(
        ["git", "init", "--bare", "--quiet", str(foreign)], check=True, env=clean_env
    )
    original_config = (foreign / "config").read_bytes()
    original_head = (foreign / "HEAD").read_bytes()
    monkeypatch.setenv("GIT_DIR", str(foreign))
    monkeypatch.setenv("GIT_INDEX_FILE", str(foreign / "hook-index"))

    root = _source_checkout(tmp_path, monkeypatch)

    assert (foreign / "config").read_bytes() == original_config
    assert (foreign / "HEAD").read_bytes() == original_head
    assert not (foreign / "hook-index").exists()
    expected = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], env=clean_env, text=True
    ).strip()
    assert capture_environment()["git_revision"] == expected
    assert capture_environment()["git_dirty"] == "false"


def test_records_actual_source_revision_and_dirty_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _source_checkout(tmp_path, monkeypatch)
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
        env={
            key: value
            for key, value in os.environ.items()
            if not key.startswith("GIT_")
        },
    ).strip()
    env = capture_environment()
    assert env["git_revision"] == revision
    assert env["git_dirty"] == "false"
    (root / "untracked.txt").write_text("dirty", encoding="utf-8")
    assert capture_environment()["git_dirty"] == "true"


def test_installed_package_does_not_report_current_workspace_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from visionforge.utils import environment

    root = _source_checkout(tmp_path, monkeypatch)
    monkeypatch.chdir(root)
    monkeypatch.setattr(
        environment,
        "__file__",
        str(tmp_path / "site-packages/visionforge/utils/environment.py"),
    )
    env = capture_environment()
    assert env["git_revision"] == "unknown"
    assert env["git_dirty"] == "unknown"


def test_git_failure_is_bounded_and_does_not_break_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from visionforge.utils import environment

    _source_checkout(tmp_path, monkeypatch)

    def unavailable(*args: object, **kwargs: object) -> None:
        assert kwargs["timeout"] == 2
        raise subprocess.TimeoutExpired("git", 2)

    monkeypatch.setattr(environment.subprocess, "run", unavailable)
    env = capture_environment()
    assert env["git_revision"] == "unknown"
    assert env["git_dirty"] == "unknown"


def test_import_never_invokes_git(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    from visionforge.utils import environment

    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: calls.append(args))
    importlib.reload(environment)
    assert calls == []


def test_source_layout_without_own_git_does_not_use_ancestor_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from visionforge.utils import environment

    root = _source_checkout(tmp_path, monkeypatch)
    nested = root / "nested/src/visionforge/utils/environment.py"
    nested.parent.mkdir(parents=True)
    nested.write_text("", encoding="utf-8")
    (root / "nested/pyproject.toml").write_text(
        '[project]\nname="visionforge-studio"\n'
    )
    monkeypatch.setattr(environment, "__file__", str(nested))
    assert capture_environment()["git_revision"] == "unknown"


def test_source_identity_ignores_git_environment_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _source_checkout(tmp_path, monkeypatch)
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "unrelated.git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path))
    assert capture_environment()["git_revision"] != "unknown"
    assert capture_environment()["git_dirty"] == "false"
    assert root.is_dir()
