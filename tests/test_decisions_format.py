"""Every ADR in docs/dev/DECISIONS.md has a numbered heading and a valid status."""

from __future__ import annotations

import re
from pathlib import Path

ENCODING = "utf-8"
ROOT = Path(__file__).resolve().parents[1]
DECISIONS = ROOT / "docs" / "dev" / "DECISIONS.md"
TEMPLATE = ROOT / "docs" / "dev" / "ADR-TEMPLATE.md"
CONTRIBUTING = ROOT / "CONTRIBUTING.md"

HEADING = re.compile(r"^## ADR-(\d{3,}) — \S", re.MULTILINE)
STATUS = re.compile(r"^\*\*Status:\*\*[ \t]*(.*)$", re.MULTILINE)
VALID_STATUS = re.compile(
    r"^(Proposed|Accepted|Rejected|"
    r"Superseded by ADR-\d+|Superseded in part by ADR-\d+|Amended by ADR-\d+)\b"
)


def _entries() -> list[tuple[str, str]]:
    """Return (heading line, body) for each `## ` section after the preamble."""
    text = DECISIONS.read_text(encoding=ENCODING)
    sections = re.split(r"(?m)^## ", text)[1:]
    return [("## " + s.splitlines()[0], s) for s in sections]


def test_every_entry_has_a_numbered_heading() -> None:
    bad = [h for h, _ in _entries() if not HEADING.match(h)]
    assert bad == []


def test_adr_numbers_are_unique() -> None:
    numbers = HEADING.findall(DECISIONS.read_text(encoding=ENCODING))
    assert len(numbers) == len(set(numbers))
    assert len(numbers) >= 184


def test_every_entry_has_a_status_with_a_valid_prefix() -> None:
    bad: list[str] = []
    for heading, body in _entries():
        match = STATUS.search(body)
        if match is None or not VALID_STATUS.match(match.group(1).strip()):
            bad.append(heading)
    assert bad == []


def test_status_check_rejects_unknown_prefixes() -> None:
    assert VALID_STATUS.match("Accepted — shipped 2026-08-05")
    assert VALID_STATUS.match("Superseded in part by ADR-060")
    assert VALID_STATUS.match("Amended by ADR-012")
    assert not VALID_STATUS.match("Maybe")
    assert not VALID_STATUS.match("Superseded by ADR-x")


def test_preamble_lists_the_valid_statuses() -> None:
    head = "\n".join(DECISIONS.read_text(encoding=ENCODING).splitlines()[:40])
    for status in (
        "Proposed",
        "Accepted",
        "Superseded by",
        "Superseded in part by",
        "Amended by",
        "Rejected",
    ):
        assert status in head


def test_template_exists_and_contributing_links_it() -> None:
    assert TEMPLATE.is_file()
    assert "ADR-TEMPLATE.md" in CONTRIBUTING.read_text(encoding=ENCODING)
