"""Model-card extraction preserves the pre-refactor PT/EN output byte for byte."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent
from typing import Literal

import pytest

from visionforge.gui.api.model_card import render_run_markdown
from visionforge.gui.api.routes import _render_run_markdown
from visionforge.utils.messages import using_lang

FIXTURES = Path(__file__).parent / "fixtures" / "model_cards"


@pytest.mark.parametrize(
    "source", sorted(FIXTURES.glob("*.json")), ids=lambda p: p.stem
)
@pytest.mark.parametrize("lang", ["pt", "en"])
def test_preserves_complete_model_card(source: Path, lang: Literal["pt", "en"]) -> None:
    """Keep text, ordering and missing-cell semantics independent of the renderer."""
    data = json.loads(source.read_text(encoding="utf-8"))
    original = copy.deepcopy(data)
    with using_lang(lang):
        result = _render_run_markdown(Path("fixture/run_1"), data)
    expected = source.with_suffix(f".{lang}.md").read_text(encoding="utf-8")
    # The goldens were captured on Windows; the card prints the OS path separator.
    assert result == expected.replace("fixture\\run_1", str(Path("fixture/run_1")))
    assert data == original


def test_routes_keep_the_pure_renderer_alias() -> None:
    """Existing route callers use the same renderer as direct consumers."""
    assert _render_run_markdown is render_run_markdown


def test_renderer_import_and_use_have_no_runtime_side_effects(tmp_path: Path) -> None:
    """Rendering needs no router, training state, filesystem writes or network."""
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            dedent("""\
                import asyncio
                import sys
                from pathlib import Path

                def refuse_side_effects(event, args):
                    if event in {"os.mkdir", "socket.connect", "subprocess.Popen", "sqlite3.connect"}:
                        raise AssertionError(f"Unexpected rendering side effect: {event}")
                    if event == "open" and args[1] and any(flag in args[1] for flag in "wax+"):
                        raise AssertionError(f"Unexpected file write: {args}")

                def forbidden_runtime(*args, **kwargs):
                    raise AssertionError("Rendering created a queue or task")

                asyncio.Queue = forbidden_runtime
                asyncio.create_task = forbidden_runtime
                sys.addaudithook(refuse_side_effects)
                from visionforge.gui.api.model_card import render_run_markdown
                from visionforge.utils.messages import using_lang

                with using_lang("en"):
                    assert render_run_markdown(Path("missing/run"), {}).startswith("# run\\n")
                for module in (
                    "visionforge.gui.api.routes",
                    "visionforge.gui.api.run_queue",
                    "visionforge.gui.server",
                    "visionforge.core.trainer",
                    "fastapi",
                    "torch",
                ):
                    assert module not in sys.modules, module
            """),
        ],
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"),
        },
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert list(tmp_path.iterdir()) == []
