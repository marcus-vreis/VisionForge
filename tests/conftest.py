import pytest


@pytest.fixture(scope="session")
def project_root(tmp_path_factory: pytest.TempPathFactory):
    """Session-scoped temporary directory for integration tests."""
    return tmp_path_factory.mktemp("visionforge_test_root")
