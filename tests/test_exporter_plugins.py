"""Third-party exporters: entry-point discovery, the protocol and ``visionforge export``."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest
from loguru import logger

import visionforge.core.exporter_plugins as plugins
from visionforge.__main__ import main
from visionforge.core.exporter_plugins import (
    ENTRY_POINT_GROUP,
    ExportResult,
    RunExporter,
    discover_exporters,
    run_exporter,
)


class FakeEntryPoint:
    """The part of ``importlib.metadata.EntryPoint`` that discovery uses."""

    def __init__(self, name: str, loader: Callable[[], Any]) -> None:
        self.name = name
        self._loader = loader

    def load(self) -> Any:
        return self._loader()


class FolderExporter:
    """Writes a marker file next to the run's destination folder."""

    name = "folder"
    description = "Copies the run summary to a folder"

    def __init__(self) -> None:
        self.calls: list[tuple[Path, dict[str, str]]] = []

    def export(self, run_dir: Path, options: Mapping[str, str]) -> ExportResult:
        self.calls.append((run_dir, dict(options)))
        target = Path(options["to"]) / "exported.txt"
        target.write_text(run_dir.name, encoding="utf-8")
        return ExportResult(str(target), notes=("done",))


class NoExport:
    name = "broken"
    description = "has no export()"


def _install(
    monkeypatch: pytest.MonkeyPatch, entries: list[FakeEntryPoint]
) -> list[dict[str, str]]:
    seen: list[dict[str, str]] = []

    def fake_entry_points(**selector: str) -> list[FakeEntryPoint]:
        seen.append(selector)
        return entries

    monkeypatch.setattr(plugins, "entry_points", fake_entry_points)
    return seen


@pytest.fixture
def warnings() -> Any:
    messages: list[str] = []
    handler = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    yield messages
    logger.remove(handler)


def _run_dir(tmp_path: Path) -> Path:
    run = tmp_path / "run_1"
    run.mkdir()
    (run / "run.json").write_text("{}", encoding="utf-8")
    return run


def _invoke(monkeypatch: pytest.MonkeyPatch, args: list[str]) -> int:
    monkeypatch.setattr("sys.argv", ["visionforge", *args])
    try:
        main()
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


def test_discovers_instance_and_class_under_the_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = FolderExporter()

    class Other:
        name = "other"
        description = "a class"

        def export(self, run_dir: Path, options: Mapping[str, str]) -> ExportResult:
            return ExportResult("x")

    seen = _install(
        monkeypatch,
        [FakeEntryPoint("a", lambda: instance), FakeEntryPoint("b", lambda: Other)],
    )
    found = discover_exporters()
    assert seen == [{"group": ENTRY_POINT_GROUP}]
    assert ENTRY_POINT_GROUP == "visionforge.exporters"
    assert set(found) == {"folder", "other"}
    assert found["folder"] is instance
    assert isinstance(found["other"], Other)
    assert isinstance(found["folder"], RunExporter)


def test_entry_point_that_fails_to_load_is_skipped(
    monkeypatch: pytest.MonkeyPatch, warnings: list[str]
) -> None:
    def boom() -> Any:
        raise ImportError("missing dependency")

    _install(
        monkeypatch,
        [FakeEntryPoint("bad", boom), FakeEntryPoint("ok", FolderExporter)],
    )
    assert set(discover_exporters()) == {"folder"}
    assert any("bad" in w and "missing dependency" in w for w in warnings)


def test_class_that_cannot_be_instantiated_is_skipped(
    monkeypatch: pytest.MonkeyPatch, warnings: list[str]
) -> None:
    class NeedsArgs:
        name = "needs"
        description = "d"

        def __init__(self, token: str) -> None:
            self.token = token

        def export(self, run_dir: Path, options: Mapping[str, str]) -> ExportResult:
            return ExportResult("x")

    _install(monkeypatch, [FakeEntryPoint("needs", lambda: NeedsArgs)])
    assert discover_exporters() == {}
    assert any("needs" in w for w in warnings)


def test_object_that_does_not_follow_the_protocol_is_skipped(
    monkeypatch: pytest.MonkeyPatch, warnings: list[str]
) -> None:
    class EmptyName(FolderExporter):
        name = ""

    _install(
        monkeypatch,
        [
            FakeEntryPoint("noexport", NoExport),
            FakeEntryPoint("emptyname", EmptyName),
            FakeEntryPoint("string", lambda: "not an exporter"),
        ],
    )
    assert discover_exporters() == {}
    for entry in ("noexport", "emptyname", "string"):
        assert any(entry in w for w in warnings)


def test_duplicate_name_keeps_the_first(
    monkeypatch: pytest.MonkeyPatch, warnings: list[str]
) -> None:
    first, second = FolderExporter(), FolderExporter()
    _install(
        monkeypatch,
        [FakeEntryPoint("one", lambda: first), FakeEntryPoint("two", lambda: second)],
    )
    found = discover_exporters()
    assert found == {"folder": first}
    assert found["folder"] is not second
    assert any("two" in w and "folder" in w for w in warnings)


def test_run_dir_without_run_json_is_refused(tmp_path: Path) -> None:
    exporter = FolderExporter()
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="run.json"):
        run_exporter(exporter, empty, {"to": str(tmp_path)})
    assert exporter.calls == []


def test_run_exporter_passes_options_and_returns_the_result(tmp_path: Path) -> None:
    exporter = FolderExporter()
    run = _run_dir(tmp_path)
    result = run_exporter(exporter, run, {"to": str(tmp_path)})
    assert exporter.calls == [(run, {"to": str(tmp_path)})]
    assert result == ExportResult(str(tmp_path / "exported.txt"), notes=("done",))
    assert (tmp_path / "exported.txt").read_text(encoding="utf-8") == "run_1"


def test_export_result_is_frozen() -> None:
    with pytest.raises(AttributeError):
        ExportResult("x").destination = "y"  # type: ignore[misc]


def test_cli_list_prints_name_and_description(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _install(monkeypatch, [FakeEntryPoint("a", FolderExporter)])
    assert _invoke(monkeypatch, ["export", "--list"]) == 0
    assert capsys.readouterr().out.strip() == (
        "folder\tCopies the run summary to a folder"
    )


def test_cli_list_without_plugins(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _install(monkeypatch, [])
    assert _invoke(monkeypatch, ["export", "--list"]) == 0
    assert "folder" not in capsys.readouterr().out


def test_cli_export_runs_the_exporter_and_prints_the_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run = _run_dir(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    _install(monkeypatch, [FakeEntryPoint("a", FolderExporter)])
    code = _invoke(
        monkeypatch,
        ["export", str(run), "--to", "folder", "--option", f"to={out}"],
    )
    captured = capsys.readouterr()
    assert code == 0
    assert captured.out.strip() == str(out / "exported.txt")
    assert captured.err.strip() == "done"
    assert (out / "exported.txt").read_text(encoding="utf-8") == "run_1"


def test_cli_unknown_exporter_lists_the_available_ones(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run = _run_dir(tmp_path)
    _install(monkeypatch, [FakeEntryPoint("a", FolderExporter)])
    code = _invoke(monkeypatch, ["export", str(run), "--to", "nope"])
    err = capsys.readouterr().err
    assert code == 2
    assert "nope" in err
    assert "folder" in err


def test_cli_run_dir_without_run_json_exits_2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install(monkeypatch, [FakeEntryPoint("a", FolderExporter)])
    code = _invoke(monkeypatch, ["export", str(tmp_path), "--to", "folder"])
    assert code == 2
    assert "run.json" in capsys.readouterr().err


def test_cli_bad_option_and_missing_arguments_exit_2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run = _run_dir(tmp_path)
    _install(monkeypatch, [FakeEntryPoint("a", FolderExporter)])
    assert (
        _invoke(monkeypatch, ["export", str(run), "--to", "folder", "--option", "x"])
        == 2
    )
    assert "x" in capsys.readouterr().err
    assert _invoke(monkeypatch, ["export"]) == 2
