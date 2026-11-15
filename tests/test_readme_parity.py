"""README.md must cover every CLI subcommand and keep English and Portuguese in parity."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_README = _ROOT / "README.md"
_INSTALL = _ROOT / "docs" / "INSTALL.md"
_CLI_MAIN = _ROOT / "src" / "visionforge" / "__main__.py"
_PT_HEADER = "# Português (pt-BR)"


def _readme_halves() -> tuple[str, str]:
    text = _README.read_text(encoding="utf-8")
    assert _PT_HEADER in text
    en, pt = text.split(_PT_HEADER, 1)
    return en, pt


def _cli_subcommands() -> list[str]:
    tree = ast.parse(_CLI_MAIN.read_text(encoding="utf-8"))
    commands: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "subparsers"
            and node.func.attr == "add_parser"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            commands.append(node.args[0].value)
    return commands


def test_cli_subcommands_extracted_from_argparse() -> None:
    commands = _cli_subcommands()
    assert commands == [
        "run",
        "gui",
        "doctor",
        "selftest",
        "new-task",
        "audit-data",
        "bundle",
        "verify-bundle",
    ]


@pytest.mark.parametrize("subcommand", _cli_subcommands())
def test_every_cli_subcommand_is_documented_in_en_and_pt(subcommand: str) -> None:
    en, pt = _readme_halves()
    token = f"visionforge {subcommand}"
    assert token in en, f"Missing '{token}' in English README"
    assert token in pt, f"Missing '{token}' in Portuguese README"


def test_en_and_pt_have_matching_h2_sections() -> None:
    en, pt = _readme_halves()
    en_h2 = re.findall(r"^## .+$", en, re.MULTILINE)
    pt_h2 = re.findall(r"^## .+$", pt, re.MULTILINE)
    assert len(en_h2) == len(pt_h2)
    assert len(en_h2) >= 10


def test_readme_and_install_cover_colab_docker_hub_code_tab_and_explorer() -> None:
    en, pt = _readme_halves()
    install = _INSTALL.read_text(encoding="utf-8")

    assert len(re.findall(r"(?i)colab", en)) >= 1
    assert len(re.findall(r"(?i)colab", pt)) >= 1
    assert len(re.findall(r"(?i)docker", en)) >= 1
    assert len(re.findall(r"(?i)docker", pt)) >= 1
    assert len(re.findall(r"(?i)docker", install)) >= 1

    assert "visionforge-studio[hub]" in en
    assert "visionforge-studio[hub]" in pt
    assert "docs/gui/codigo.md" in en
    assert "docs/gui/codigo.md" in pt
    assert "docs/gui/colab.md" in en
    assert "docs/gui/colab.md" in pt
    assert "next release" in en.lower()
    assert "próxima release" in pt.lower()
