"""Every citation in the two papers has an entry in its bibliography (#180)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
JOSS = ROOT / "paper" / "paper.md"
JOSS_BIB = ROOT / "paper" / "paper.bib"
STUDY = ROOT / "research" / "seed_variance" / "paper" / "main.tex"
STUDY_BIB = ROOT / "research" / "seed_variance" / "paper" / "refs.bib"
# The Holm DOI that resolves nowhere (doi.org and Crossref answer 404).
DEAD_HOLM_DOI = "10.2307/4615733"

_BRACKET = re.compile(r"\[[^\]]*@[^\]]*\]")
_PANDOC_KEY = re.compile(r"@([A-Za-z](?:[\w:.-]*\w)?)")
_TEX_CITE = re.compile(r"\\cite[pt]?\*?(?:\[[^\]]*\])*\{([^}]*)\}")
_BIB_KEY = re.compile(r"^@\w+\{([^,\s]+),", re.MULTILINE)
_MD_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)


def markdown_citations(text: str) -> set[str]:
    """The keys of every ``[@key; @other]`` group outside HTML comments."""
    text = _MD_COMMENT.sub("", text)
    return {
        key for group in _BRACKET.findall(text) for key in _PANDOC_KEY.findall(group)
    }


def latex_citations(text: str) -> set[str]:
    """The keys of every ``\\cite``, ``\\citep`` and ``\\citet``."""
    return {
        key.strip()
        for group in _TEX_CITE.findall(text)
        for key in group.split(",")
        if key.strip()
    }


def bib_keys(text: str) -> set[str]:
    return set(_BIB_KEY.findall(text))


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class TestParsers:
    def test_markdown_groups_and_comments(self) -> None:
        text = "a [@x; @y, p. 2] b <!-- [@hidden] --> c [see @z] d@mail.org"
        assert markdown_citations(text) == {"x", "y", "z"}

    def test_latex_variants(self) -> None:
        text = r"\citep{a,b} \citet[p.~3]{c} \cite{ d }"
        assert latex_citations(text) == {"a", "b", "c", "d"}


class TestBibliographies:
    def test_every_joss_citation_has_an_entry(self) -> None:
        cited = markdown_citations(_read(JOSS))
        assert cited
        assert sorted(cited - bib_keys(_read(JOSS_BIB))) == []

    def test_every_study_citation_has_an_entry(self) -> None:
        cited = latex_citations(_read(STUDY))
        assert cited
        assert sorted(cited - bib_keys(_read(STUDY_BIB))) == []

    @pytest.mark.parametrize("bib", [JOSS_BIB, STUDY_BIB], ids=lambda p: p.name)
    def test_no_dead_holm_doi(self, bib: Path) -> None:
        assert DEAD_HOLM_DOI not in _read(bib)

    def test_holm_is_cited_in_the_joss_paper(self) -> None:
        assert "holm1979simple" in markdown_citations(_read(JOSS))
