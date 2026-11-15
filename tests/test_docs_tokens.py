"""The tokens guide must keep up with the credential stores in the code."""

from pathlib import Path
from typing import get_args

import pytest

from visionforge.gui.api.profile_tokens import Service, token_file
from visionforge.gui.api.profiles import Profile
from visionforge.utils.credentials import PROVIDERS, config_dir

_ROOT = Path(__file__).resolve().parent.parent
_GUIDE = _ROOT / "docs" / "gui" / "tokens.md"
_SERVICES = get_args(Service)
_GUIDE_LINK = "gui/tokens.md"


@pytest.fixture(scope="module")
def guide() -> str:
    return _GUIDE.read_text(encoding="utf-8")


def test_guide_names_home_variable_and_credentials_file(guide: str):
    assert "VISIONFORGE_HOME" in guide
    assert "credentials.json" in guide


@pytest.mark.parametrize("provider", PROVIDERS)
def test_guide_covers_every_dataset_provider(guide: str, provider: str):
    assert f"`{provider}`" in guide


@pytest.mark.parametrize("service", _SERVICES)
def test_guide_documents_every_token_directory(guide: str, service: str):
    assert f"`{service}-tokens/<sha256>`" in guide


@pytest.mark.parametrize("name", ["KAGGLE_API_TOKEN", "HF_TOKEN"])
def test_guide_mentions_environment_fallbacks(guide: str, name: str):
    assert name in guide


@pytest.mark.parametrize("service", _SERVICES)
def test_token_file_matches_documented_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, service: str
):
    monkeypatch.setenv("VISIONFORGE_HOME", str(tmp_path / "home"))
    profile = Profile(slug="lab", name="Lab", models_dir=tmp_path / "models")
    path = token_file(service, profile)  # type: ignore[arg-type]
    assert config_dir() == tmp_path / "home"
    assert path.parent == tmp_path / "home" / f"{service}-tokens"
    assert len(path.name) == 64


def test_default_config_dir_is_documented_home(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("VISIONFORGE_HOME", raising=False)
    assert config_dir() == Path.home() / ".visionforge"


def test_readme_links_to_guide():
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    assert _GUIDE_LINK in readme


def test_datasets_guide_links_to_guide():
    datasets = (_ROOT / "docs" / "DATASETS.md").read_text(encoding="utf-8")
    assert _GUIDE_LINK in datasets
