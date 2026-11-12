"""Each exporter has a short page in docs/gui/ that states its limits (#213)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GUI_DOCS = ROOT / "docs" / "gui"
ROUTES = ROOT / "src" / "visionforge" / "gui" / "api"

# service -> (route module that exports it, ADR that records the decision)
EXPORTERS = {
    "mlflow": ("routes.py", "ADR-170"),
    "wandb": ("wandb_routes.py", "ADR-172"),
    "comet": ("comet_routes.py", "ADR-200"),
    "zenodo": ("zenodo_routes.py", "ADR-174"),
    "github": ("github_routes.py", "ADR-175"),
    "dvc": ("dvc_routes.py", "ADR-176"),
}
NOT_SENT_HEADING = re.compile(r"^## O que não é enviado$", re.MULTILINE)
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)[^)]*\)")


def _page(service: str) -> Path:
    return GUI_DOCS / f"{service}.md"


@pytest.mark.parametrize("service", EXPORTERS)
def test_route_module_exists(service: str) -> None:
    assert (ROUTES / EXPORTERS[service][0]).is_file()


@pytest.mark.parametrize("service", EXPORTERS)
def test_page_exists(service: str) -> None:
    assert _page(service).is_file()


@pytest.mark.parametrize("service", EXPORTERS)
def test_page_cites_its_adr(service: str) -> None:
    text = _page(service).read_text(encoding="utf-8")
    adr = EXPORTERS[service][1]
    assert re.search(rf"\b{adr}\b", text)


@pytest.mark.parametrize("service", EXPORTERS)
def test_page_has_not_sent_section(service: str) -> None:
    text = _page(service).read_text(encoding="utf-8")
    assert NOT_SENT_HEADING.search(text)


@pytest.mark.parametrize("service", EXPORTERS)
def test_page_has_no_relative_link(service: str) -> None:
    text = _page(service).read_text(encoding="utf-8")
    targets = MARKDOWN_LINK.findall(text)
    relative = [t for t in targets if not t.startswith(("http://", "https://", "#"))]
    assert relative == []


@pytest.mark.parametrize("service", ["wandb", "comet", "zenodo", "github"])
def test_token_page_links_to_tokens_guide(service: str) -> None:
    text = _page(service).read_text(encoding="utf-8")
    assert (
        "https://github.com/marcus-vreis/VisionForge/blob/development/docs/gui/tokens.md"
        in text
    )
