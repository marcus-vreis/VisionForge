"""Every hook in .pre-commit-config.yaml is described in CONTRIBUTING.md."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
HOOKS_HEADING = "## Pre-commit hooks"


def _configured_hook_ids() -> list[str]:
    config = yaml.safe_load(
        (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    )
    return [hook["id"] for repo in config["repos"] for hook in repo["hooks"]]


def _documented_hook_ids() -> set[str]:
    text = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    section = text.split(HOOKS_HEADING, 1)[1].split("\n## ", 1)[0]
    ids: set[str] = set()
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        ids.add(line.strip("|").split("|")[0].strip().strip("`"))
    return ids


def test_every_precommit_hook_is_in_the_contributing_table() -> None:
    documented = _documented_hook_ids()
    missing = [hook for hook in _configured_hook_ids() if hook not in documented]
    assert missing == []


def test_configured_hooks_are_found() -> None:
    assert "codespell" in _configured_hook_ids()


def _hook_stages(hook_id: str) -> list[str] | None:
    config = yaml.safe_load(
        (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    )
    for repo in config["repos"]:
        for hook in repo["hooks"]:
            if hook["id"] == hook_id:
                stages: list[str] | None = hook.get("stages")
                return stages
    raise AssertionError(f"hook {hook_id} not configured")


def test_pytest_runs_on_push_and_the_quick_hooks_on_commit() -> None:
    assert _hook_stages("pytest") == ["pre-push"]
    for quick in ("ruff", "ruff-format", "codespell", "mypy"):
        assert _hook_stages(quick) is None


def test_contributing_documents_installing_both_stages() -> None:
    text = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    assert "pre-commit install --hook-type pre-commit --hook-type pre-push" in text
    assert "stays quick" not in text
    assert "(ADR-186)" in text
