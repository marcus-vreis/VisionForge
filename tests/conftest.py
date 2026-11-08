import pytest


@pytest.fixture(scope="session")
def project_root(tmp_path_factory: pytest.TempPathFactory):
    """Session-scoped temporary directory for integration tests."""
    return tmp_path_factory.mktemp("visionforge_test_root")


@pytest.fixture
def fake_mlflow(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    """Install the in-memory MLflow fake as the ``mlflow`` module (ADR-170)."""
    from .fake_mlflow import install_fake_mlflow

    return install_fake_mlflow(monkeypatch)
