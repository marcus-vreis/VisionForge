"""
Shared pytest fixtures for the VisionForge test suite.

Fixtures defined here are automatically available to all tests
without requiring explicit imports.
"""

import pytest


@pytest.fixture(scope="session")
def project_root(tmp_path_factory: pytest.TempPathFactory):
    """Return a session-scoped temporary directory for integration tests."""
    return tmp_path_factory.mktemp("visionforge_test_root")
