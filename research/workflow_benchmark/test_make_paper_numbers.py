from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from research.workflow_benchmark import export_results, make_paper_numbers
from research.workflow_benchmark.make_paper_numbers import (
    PAPER,
    RESULTS,
    normalized,
    paper_phrases,
    protocol_margins,
    ratio_interval,
    t_interval,
)

SECTION_START = "A layer over the training libraries"
SECTION_END = "not shown to be within 5% for every recipe."
DECIMAL = re.compile(r"[+-]?\d+\.\d+")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
HOME_PATH = re.compile(r"Users(?:\\\\|\\|/)+(?!Public)\w+|/home/\w+", re.IGNORECASE)


def _benchmark_section() -> str:
    paper = normalized(PAPER.read_text(encoding="utf-8"))
    return paper[paper.index(SECTION_START) : paper.index(SECTION_END)]


@pytest.fixture(scope="module")
def phrases() -> list[str]:
    return paper_phrases()


class TestPaperNumbers:
    def test_every_phrase_is_printed_in_the_paper(self, phrases: list[str]) -> None:
        paper = normalized(PAPER.read_text(encoding="utf-8"))
        assert [p for p in phrases if p not in paper] == []

    def test_every_decimal_of_the_section_is_recomputed(
        self, phrases: list[str]
    ) -> None:
        recomputed = set(DECIMAL.findall(" ".join(phrases)))
        cited = DECIMAL.findall(_benchmark_section())
        assert len(cited) == 62
        assert [x for x in cited if x not in recomputed] == []

    def test_main_exits_zero_on_the_paper(self) -> None:
        assert make_paper_numbers.main() == 0

    def test_main_exits_one_when_a_number_is_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paper = tmp_path / "paper.md"
        paper.write_text("no numbers here", encoding="utf-8")
        monkeypatch.setattr(make_paper_numbers, "PAPER", paper)
        assert make_paper_numbers.main() == 1


class TestAgainstTheOriginalSummaries:
    """The recomputation agrees with what the series' own scripts wrote."""

    def test_real_v1(self) -> None:
        summary = json.loads((RESULTS / "real_summary.json").read_text("utf-8"))
        cls = summary["cls"]
        mean, low, high = t_interval(
            [a - b for a, b in zip(cls["vf"], cls["native"], strict=True)], 0.95
        )
        assert mean == pytest.approx(cls["diff_mean"], rel=1e-12)
        assert [low, high] == pytest.approx(cls["diff_ci95"], rel=1e-12)
        det = summary["det"]["time_with_reruns"]
        ratio, low, high = ratio_interval(det["per_seed"])
        assert [ratio, low, high] == pytest.approx(
            [det["mean"], det["ci90_low"], det["ci90_high"]], rel=1e-12
        )

    def test_gui_v2(self) -> None:
        summary = json.loads((RESULTS / "gui_v2_summary.json").read_text("utf-8"))
        cls = summary["cls"]
        ratio, low, high = ratio_interval(cls["ratio_per_seed"])
        assert [ratio, low, high] == pytest.approx(
            [cls["ratio"], *cls["ratio_ci90"]], rel=1e-12
        )


class TestProtocolMargins:
    def test_reads_the_frozen_margins(self) -> None:
        assert protocol_margins(RESULTS / "gui-v1") == ("0.005", "5")

    def test_refuses_a_protocol_without_margins(self, tmp_path: Path) -> None:
        (tmp_path / "PROTOCOL.md").write_text("nothing", encoding="utf-8")
        with pytest.raises(ValueError, match="no margins"):
            protocol_margins(tmp_path)


class TestPublishedFiles:
    def test_no_home_path_username_or_email(self) -> None:
        offenders = []
        for path in sorted(RESULTS.rglob("*")):
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8")
            if HOME_PATH.search(text) or EMAIL.search(text):
                offenders.append(str(path.relative_to(RESULTS)))
            elif "marcus" in text.lower():
                offenders.append(str(path.relative_to(RESULTS)))
        assert offenders == []

    def test_published_results_stay_small(self) -> None:
        total = sum(p.stat().st_size for p in RESULTS.rglob("*") if p.is_file())
        assert total <= 1_500_000


class TestExport:
    def test_scrub_replaces_the_home_folder_only(self) -> None:
        text = (
            'a "C:\\\\Users\\\\someone\\\\vfsmoke\\\\x" b C:/Users/someone/data '
            "c /c/Users/someone/venv d /home/someone/e f C:/Program Files/x"
        )
        assert export_results.scrub(text) == (
            'a "~\\\\vfsmoke\\\\x" b ~/data c ~/venv d ~/e f C:/Program Files/x'
        )

    def test_export_copies_text_and_skips_images(self, tmp_path: Path) -> None:
        source = tmp_path / "src"
        for series in export_results.SERIES:
            (source / series).mkdir(parents=True)
        (source / "real-v1" / "summary.log").write_text(
            "B=/c/Users/someone/bench\n", encoding="utf-8"
        )
        (source / "gui-v1" / "end.png").write_bytes(b"\x89PNG")
        (source / "real_summary.md").write_text("# r\n", encoding="utf-8")
        target = tmp_path / "out"
        assert export_results.export(source, target) == 2
        log = (target / "real-v1" / "summary.log").read_text(encoding="utf-8")
        assert log == "B=~/bench\n"
        assert not (target / "gui-v1" / "end.png").exists()

    def test_main_refuses_a_source_without_the_series(self, tmp_path: Path) -> None:
        assert export_results.main(["--source", str(tmp_path)]) == 1
