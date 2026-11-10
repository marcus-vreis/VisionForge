import os

import pytest

from visionforge.core.dvc_export import _GIT_LOCAL_ENV


@pytest.fixture(scope="session")
def project_root(tmp_path_factory: pytest.TempPathFactory):
    """Session-scoped temporary directory for integration tests."""
    return tmp_path_factory.mktemp("visionforge_test_root")


@pytest.fixture
def fake_mlflow(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    """Install the in-memory MLflow fake as the ``mlflow`` module (ADR-170)."""
    from .fake_mlflow import install_fake_mlflow

    return install_fake_mlflow(monkeypatch)


@pytest.fixture
def clean_git_env(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Drop the git variables a hook exports, so a test's git stays in its tmp_path.

    A test that runs ``git init`` while ``GIT_DIR`` is inherited from a commit hook
    reinitializes the real repository. Uses the list production strips in
    ``run_command``; returns the clean environment for ``subprocess`` calls.
    """
    for name in list(os.environ):
        if name.upper() in _GIT_LOCAL_ENV:
            monkeypatch.delenv(name)
    return dict(os.environ)
