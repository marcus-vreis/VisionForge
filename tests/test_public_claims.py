"""Public privacy and positioning claims must match what the software does."""

from __future__ import annotations

from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_README = _ROOT / "README.md"
_INSTALL = _ROOT / "docs" / "INSTALL.md"
_PYPROJECT = _ROOT / "pyproject.toml"
_CITATION = _ROOT / "CITATION.cff"

_FORBIDDEN_CLAIMS = (
    "no cloud",
    "sem nuvem",
    "nothing leaves the machine",
)

_PT_OPT_IN_CLAIM = (
    "roda na sua máquina; nada sai dela sem você pedir "
    "(datasets, Colab e exportadores são opt-in, com prévia local)"
)
_EN_OPT_IN_CLAIM = (
    "runs on your machine; nothing leaves it unless you ask for it "
    "(datasets, Colab and exporters are opt-in, with a local preview)"
)


@pytest.mark.parametrize("doc_path", [_README, _INSTALL], ids=["README", "INSTALL"])
@pytest.mark.parametrize("claim", _FORBIDDEN_CLAIMS)
def test_docs_do_not_make_unqualified_no_cloud_claims(
    doc_path: Path, claim: str
) -> None:
    text = doc_path.read_text(encoding="utf-8").lower()
    assert claim not in text


def test_local_first_positioning_is_consistent_across_metadata() -> None:
    readme = _README.read_text(encoding="utf-8").lower()
    pyproject_line_6 = _PYPROJECT.read_text(encoding="utf-8").splitlines()[5].lower()
    citation = _CITATION.read_text(encoding="utf-8").lower()

    assert "local-first" in readme
    assert "local-first" in pyproject_line_6
    assert "local-first" in citation


def test_readme_and_install_state_opt_in_network_boundary() -> None:
    readme = _README.read_text(encoding="utf-8")
    install = _INSTALL.read_text(encoding="utf-8")

    assert _EN_OPT_IN_CLAIM in readme
    assert _PT_OPT_IN_CLAIM in readme
    assert " ".join(_PT_OPT_IN_CLAIM.split()) in " ".join(install.split())
