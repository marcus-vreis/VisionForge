"""Evidence commands run real audits and bundles without training."""

import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

from visionforge.__main__ import main


def _invoke(monkeypatch: pytest.MonkeyPatch, args: list[str]) -> int:
    monkeypatch.setattr("sys.argv", ["visionforge", *args])
    try:
        main()
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


def _dataset(root: Path) -> Path:
    for index, split in enumerate(("train", "val", "test")):
        folder = root / split / "a"
        folder.mkdir(parents=True)
        Image.new("RGB", (8, 8), (index * 70, 10, 20)).save(folder / "sample.png")
    return root


def test_audit_command_outputs_json_and_reports_duplicates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = _dataset(tmp_path / "dataset")
    shutil.copyfile(dataset / "train/a/sample.png", dataset / "test/a/sample.png")
    report_path = tmp_path / "report.json"
    code = _invoke(
        monkeypatch, ["audit-data", str(dataset), "--json", str(report_path)]
    )
    report = json.loads(capsys.readouterr().out)
    assert code == 1
    assert report["ok"] is False
    assert report["duplicate_groups"][0]["splits"] == ["test", "train"]
    assert json.loads(report_path.read_text(encoding="utf-8")) == report


def test_audit_custom_splits_and_invalid_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = _dataset(tmp_path / "dataset")
    (dataset / "val").rename(dataset / "validation")
    args = [
        "audit-data",
        str(dataset),
        "--split",
        "train=train",
        "--split",
        "val=validation",
        "--split",
        "test=test",
    ]
    assert _invoke(monkeypatch, args) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    assert _invoke(monkeypatch, ["audit-data", str(dataset), "--split", "invalid"]) == 2
    assert "NAME=PATH" in capsys.readouterr().err


def test_bundle_export_verify_and_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "run.json").write_text(
        '{"artifacts":{"model":"model.pth"}}', encoding="utf-8"
    )
    (run / "model.pth").write_bytes(b"checkpoint")
    destination = tmp_path / "evidence.zip"
    assert _invoke(monkeypatch, ["bundle", str(run), str(destination)]) == 0
    assert json.loads(capsys.readouterr().out)["bundle"] == str(destination)
    assert _invoke(monkeypatch, ["verify-bundle", str(destination)]) == 0
    assert isinstance(json.loads(capsys.readouterr().out), dict)
    assert _invoke(monkeypatch, ["bundle", str(run), str(destination)]) == 2
    assert capsys.readouterr().err
    destination.write_bytes(b"damaged archive")
    assert _invoke(monkeypatch, ["verify-bundle", str(destination)]) == 2
    assert capsys.readouterr().err


def test_audit_does_not_overwrite_an_existing_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _dataset(tmp_path / "dataset")
    target = tmp_path / "existing.json"
    target.write_text("KEEP", encoding="utf-8")
    assert (
        _invoke(monkeypatch, ["audit-data", str(dataset), "--json", str(target)]) == 2
    )
    assert target.read_text(encoding="utf-8") == "KEEP"
