"""Ratchet on the size of modules being split up (#204).

``gui/api/routes.py`` is broken up one domain per PR (evaluation, then the queue
and run state). Each PR that shrinks it lowers its ceiling here to the new
``wc -l`` (plus a few lines for PRs merged meanwhile), so the file cannot grow
back. Raising a ceiling needs a reason in the PR, not a new number.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Path from the repository root -> the most lines it may have (``wc -l``).
CEILINGS = {
    "src/visionforge/gui/api/routes.py": 3210,
}


def _line_count(path: Path) -> int:
    """Count newlines, which is what ``wc -l`` reports."""
    return path.read_bytes().count(b"\n")


@pytest.mark.parametrize(("relative", "ceiling"), sorted(CEILINGS.items()))
def test_module_does_not_grow_past_its_ceiling(relative: str, ceiling: int) -> None:
    lines = _line_count(ROOT / relative)
    assert lines <= ceiling, (
        f"{relative} has {lines} lines, above its ceiling of {ceiling}: move the "
        "new code to a module of its own domain (see #204)"
    )


def test_line_count_matches_wc(tmp_path: Path) -> None:
    sample = tmp_path / "sample.py"
    sample.write_bytes(b"a = 1\r\nb = 2\n\nc = 3")
    assert _line_count(sample) == 3
