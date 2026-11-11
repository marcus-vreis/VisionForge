"""README.md is published to PyPI as long_description; its links and examples must work there."""

from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_README = _ROOT / "README.md"
_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)\)")
_GITHUB_BLOB_RE = re.compile(
    r"^https://github\.com/marcus-vreis/VisionForge/(?:blob|tree)/main/([^\s)#]+)"
)


def _readme_text() -> str:
    return _README.read_text(encoding="utf-8")


def test_readme_contains_no_relative_links_for_pypi() -> None:
    text = _readme_text()
    assert "](docs/" not in text
    targets = _LINK_RE.findall(text)
    assert targets
    relative = [
        t for t in targets if not t.startswith("https://") and not t.startswith("#")
    ]
    assert relative == []


def test_readme_github_file_links_point_to_existing_paths() -> None:
    text = _readme_text()
    checked = 0
    for target in _LINK_RE.findall(text):
        match = _GITHUB_BLOB_RE.match(target)
        if match is None:
            continue
        rel_path = match.group(1)
        assert (_ROOT / rel_path).exists(), f"Missing path in README link: {rel_path}"
        checked += 1
    assert checked >= 10


def test_readme_cli_run_example_uses_exported_yaml_flow() -> None:
    text = _readme_text()
    assert "visionforge run configs/baseline.yaml" not in text
    assert text.count("visionforge run experiment.yaml") >= 2


def test_readme_status_limits_match_parallel_colab_and_roboflow_detection() -> None:
    text = _readme_text()
    assert "One-click dataset download covers classification only." not in text
    assert "O download em um clique cobre só classificação." not in text
    assert "One local training at a time" in text
    assert "Um treino local por vez" in text


def test_readme_fenced_blocks_are_balanced_for_long_description() -> None:
    text = _readme_text()
    fence_lines = [
        line for line in text.splitlines() if line.lstrip().startswith("```")
    ]
    assert len(fence_lines) >= 10
    assert len(fence_lines) % 2 == 0
