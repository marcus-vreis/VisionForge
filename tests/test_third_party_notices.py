"""THIRD_PARTY_NOTICES.md and the license / privacy sections of the docs (#176)."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_NOTICES = _ROOT / "THIRD_PARTY_NOTICES.md"
_STATIC = _ROOT / "src" / "visionforge" / "gui" / "static"
_DOCS = ("README.md", "docs/INSTALL.md")
_NOTICES_URL = (
    "https://github.com/marcus-vreis/VisionForge/blob/main/THIRD_PARTY_NOTICES.md"
)
# Font file names start with the Fontsource family slug.
_FONT_SLUGS = ("bricolage-grotesque", "familjen-grotesk", "ibm-plex-mono")


def _text(relative: str) -> str:
    return (_ROOT / relative).read_text(encoding="utf-8")


def test_wheel_metadata_names_the_notices_as_a_license_file() -> None:
    project = tomllib.loads(_text("pyproject.toml"))["project"]
    assert "THIRD_PARTY_NOTICES.md" in project["license-files"]
    assert project["license"] == "MIT"


def test_notices_name_ultralytics_as_agpl() -> None:
    notices = _NOTICES.read_text(encoding="utf-8")
    assert re.search(r"\| `ultralytics` \| AGPL-3\.0 \|", notices)
    assert re.search(r"\| `ultralytics-thop` \| AGPL-3\.0 \|", notices)
    assert re.search(r"\| `ultralytics-platform` \| AGPL-3\.0-only \|", notices)


def test_every_bundled_font_is_covered_by_the_notices() -> None:
    notices = _NOTICES.read_text(encoding="utf-8")
    fonts = sorted(_STATIC.rglob("*.woff*"))
    assert fonts, "the SPA bundle ships no font files"
    unknown = [f.name for f in fonts if not f.name.startswith(_FONT_SLUGS)]
    assert unknown == []
    for slug in _FONT_SLUGS:
        assert f"/{slug}`" in notices
    assert "SIL OPEN FONT LICENSE Version 1.1" in notices


@pytest.mark.parametrize("document", _DOCS)
def test_docs_mention_agpl_and_the_update_check_switch(document: str) -> None:
    text = _text(document)
    assert "AGPL" in text
    assert "VISIONFORGE_NO_UPDATE_CHECK" in text


@pytest.mark.parametrize("document", _DOCS)
def test_docs_link_the_notices_with_an_absolute_url(document: str) -> None:
    # PyPI renders the README outside the repository: a relative link 404s there.
    text = _text(document)
    assert _NOTICES_URL in text
    assert "](THIRD_PARTY_NOTICES.md)" not in text
    assert "](../THIRD_PARTY_NOTICES.md)" not in text


@pytest.mark.parametrize(
    "heading",
    [
        "## Third-party licenses",
        "## Network and privacy",
        "## Licenças de terceiros",
        "## Rede e privacidade",
    ],
)
def test_readme_has_both_language_sections(heading: str) -> None:
    assert heading in _text("README.md")
