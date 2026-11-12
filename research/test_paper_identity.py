"""One author identity in git and one affiliation in the paper's metadata (#231)."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CANONICAL = "Marcus Vinícius Reis <marcusvinicius0083@gmail.com>"
UFSCAR = "Universidade Federal de São Carlos (UFSCar)"


@pytest.mark.parametrize(
    "identity",
    [
        "Marcus Vinicius <marcusvinicius0083@gmail.com>",
        "Allinvila <marcus.diniz@pixforce.ai>",
    ],
)
def test_mailmap_maps_every_commit_identity_to_one(identity: str) -> None:
    mapped = subprocess.run(
        ["git", "-c", "core.quotePath=false", "check-mailmap", identity],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    ).stdout.strip()
    assert mapped == CANONICAL


def test_paper_and_citation_name_only_ufscar() -> None:
    paper = (ROOT / "paper" / "paper.md").read_text(encoding="utf-8")
    front_matter = paper.split("\n---\n", 1)[0]
    affiliations = front_matter.split("affiliations:", 1)[1]
    cff = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert re.findall(r"^\s+- name: (.+)$", affiliations, re.MULTILINE) == [UFSCAR]
    assert re.findall(r'^\s+affiliation: "(.+)"$', cff, re.MULTILINE) == [UFSCAR]
