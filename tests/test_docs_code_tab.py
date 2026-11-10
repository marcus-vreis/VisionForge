"""The Code tab and Colab guides must keep up with the tours and the Colab routes."""

import inspect
import re
import textwrap
from pathlib import Path

import pytest

from visionforge.core import code_tour
from visionforge.core.code_tour import TOURS
from visionforge.gui.api import colab_connection
from visionforge.gui.api.profiles import PROFILE_HEADER
from visionforge.gui.api.routes import router

_ROOT = Path(__file__).resolve().parent.parent
_CODE_GUIDE = _ROOT / "docs" / "gui" / "codigo.md"
_COLAB_GUIDE = _ROOT / "docs" / "gui" / "colab.md"
_DECISIONS = _ROOT / "docs" / "dev" / "DECISIONS.md"
_SECTIONS = [(task, section) for task, tour in TOURS.items() for section in tour]
_SECTION_IDS = [f"{task}-{section.id}" for task, section in _SECTIONS]
_PYTHON_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _colab_routes() -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for route in router.routes:
        path = getattr(route, "path", "")
        methods = getattr(route, "methods", None) or set()
        if "/colab/" in path:
            found.extend((method, path) for method in methods)
    return sorted(found)


def _task_part(guide: str, task: str) -> str:
    """The text under the ``##`` heading that names ``task`` in backticks."""
    parts = re.split(r"^## ", guide, flags=re.MULTILINE)
    matching = [part for part in parts if part.split("\n", 1)[0].count(f"`{task}`")]
    assert len(matching) == 1, f"no single '## ... `{task}`' heading in the guide"
    return matching[0]


@pytest.fixture(scope="module")
def code_guide() -> str:
    return _CODE_GUIDE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def colab_guide() -> str:
    return _COLAB_GUIDE.read_text(encoding="utf-8")


def test_code_guide_has_a_heading_for_every_task(code_guide: str):
    for task in TOURS:
        _task_part(code_guide, task)


@pytest.mark.parametrize(("task", "section"), _SECTIONS, ids=_SECTION_IDS)
def test_code_guide_cites_each_function_of_each_tour(
    code_guide: str, task: str, section: code_tour.Section
):
    target = section.target()
    assert callable(target)
    assert f"`{target.__qualname__}`" in _task_part(code_guide, task)


def _contains(source: str, wanted: list[str]) -> bool:
    """Whether ``wanted`` is a run of ``source`` lines, at any one indentation."""
    lines = source.split("\n")
    for start, line in enumerate(lines):
        offset = len(line) - len(line.lstrip())
        window = lines[start : start + len(wanted)]
        if len(window) == len(wanted) and all(
            got[offset:].rstrip() == want.rstrip()
            for got, want in zip(window, wanted, strict=True)
        ):
            return True
    return False


def test_code_guide_python_blocks_are_in_the_source(code_guide: str):
    sources = [inspect.getsource(code_tour)]
    sources += [inspect.getsource(section.target()) for _, section in _SECTIONS]
    blocks = _PYTHON_BLOCK.findall(code_guide)
    assert blocks, "the guide shows no code"
    for block in blocks:
        wanted = textwrap.dedent(block).strip("\n").split("\n")
        found = any(_contains(source, wanted) for source in sources)
        assert found, f"not in the source:\n{block}"


def test_code_guide_cites_the_route_and_adrs(code_guide: str):
    assert "/api/code-tour/{task}" in code_guide
    assert "/api/code-tour`" in code_guide
    assert "ADR-173" in code_guide
    assert "ADR-181" in code_guide


def test_colab_routes_were_found():
    assert ("POST", "/api/colab/login/start") in _colab_routes()


@pytest.mark.parametrize(("method", "path"), _colab_routes())
def test_colab_guide_cites_every_route(colab_guide: str, method: str, path: str):
    assert f"`{method} {path}`" in colab_guide


def test_colab_guide_cites_the_variables_of_the_code(colab_guide: str):
    assert colab_connection.PINNED_VERSION in colab_guide
    assert PROFILE_HEADER in colab_guide
    assert "google-colab-cli" in colab_guide
    assert "token.json" in colab_guide


@pytest.mark.parametrize("adr", ["ADR-163", "ADR-164"])
def test_colab_guide_cites_adrs_that_exist(colab_guide: str, adr: str):
    assert adr in colab_guide
    assert f"## {adr} " in _DECISIONS.read_text(encoding="utf-8")
