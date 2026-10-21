"""A resumed experiment cannot silently change its recorded dataset."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import pytest
from loguru import logger

from visionforge.core.dataset_fingerprint import fingerprint_dataset
from visionforge.core.resume_provenance import validate_resume_dataset


def _run(
    tmp_path: Path, method: Literal["manifest", "content"] = "manifest"
) -> tuple[Path, Path, SimpleNamespace]:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "sample.png").write_bytes(b"original")
    run = tmp_path / "run"
    run.mkdir()
    report = {
        "dataset_fingerprint": fingerprint_dataset(dataset, method=method).to_dict()
    }
    (run / "run.json").write_text(json.dumps(report), encoding="utf-8")
    return run, dataset, SimpleNamespace(data=SimpleNamespace(base_dir=dataset))


def test_unchanged_dataset_is_accepted(tmp_path: Path) -> None:
    run, _, config = _run(tmp_path)
    validate_resume_dataset(run, config)


def test_confirmed_dataset_change_is_rejected_without_rewriting_run(
    tmp_path: Path,
) -> None:
    run, dataset, config = _run(tmp_path)
    before = (run / "run.json").read_bytes()
    (dataset / "sample.png").write_bytes(b"different and longer")
    with pytest.raises(ValueError, match="dataset.*changed"):
        validate_resume_dataset(run, config)
    assert (run / "run.json").read_bytes() == before


def test_content_fingerprint_keeps_its_stronger_check_on_resume(tmp_path: Path) -> None:
    run, dataset, config = _run(tmp_path, "content")
    (dataset / "sample.png").write_bytes(b"modified")
    with pytest.raises(ValueError, match="dataset.*changed"):
        validate_resume_dataset(run, config)


def test_legacy_missing_fingerprint_has_an_explicit_warning(tmp_path: Path) -> None:
    run, _, config = _run(tmp_path)
    (run / "run.json").write_text("{}", encoding="utf-8")
    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(str(message)), format="{message}")
    try:
        validate_resume_dataset(run, config)
    finally:
        logger.remove(sink)
    assert any("cannot verify" in message for message in messages)


def test_corrupt_existing_report_is_not_silently_accepted(tmp_path: Path) -> None:
    run, _, config = _run(tmp_path)
    (run / "run.json").write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="run.json"):
        validate_resume_dataset(run, config)


def test_outputs_inside_dataset_do_not_create_a_false_mismatch(tmp_path: Path) -> None:
    run, dataset, config = _run(tmp_path)
    config.output = SimpleNamespace(models_dir=dataset / "models")
    config.output.models_dir.mkdir()
    (config.output.models_dir / "checkpoint.pt").write_bytes(b"new output")
    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(str(message)), format="{message}")
    try:
        validate_resume_dataset(run, config)
    finally:
        logger.remove(sink)
    assert any("output directories" in message for message in messages)


@pytest.mark.parametrize("payload", ["[]", "null"])
def test_nonobject_report_is_rejected(tmp_path: Path, payload: str) -> None:
    run, _, config = _run(tmp_path)
    (run / "run.json").write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="run.json"):
        validate_resume_dataset(run, config)


@pytest.mark.parametrize("kind", ["absent", "legacy", "unavailable"])
def test_unverifiable_identity_warns(tmp_path: Path, kind: str) -> None:
    run, dataset, config = _run(tmp_path)
    if kind == "absent":
        (run / "run.json").unlink()
    elif kind == "legacy":
        report = json.loads((run / "run.json").read_text(encoding="utf-8"))
        report["dataset_fingerprint"].pop("encoding_version")
        (run / "run.json").write_text(json.dumps(report), encoding="utf-8")
    else:
        config.data.base_dir = dataset / "missing"
    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(str(message)), format="{message}")
    try:
        validate_resume_dataset(run, config)
    finally:
        logger.remove(sink)
    assert any("cannot verify" in message for message in messages)
