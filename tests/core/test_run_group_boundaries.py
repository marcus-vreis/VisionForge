"""Group writes reject paths outside the selected model root before changing files."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from visionforge.core.run_groups import group_run_dir, write_run_group


def _run(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "run.json").write_text('{"sentinel": true}', encoding="utf-8")
    return folder


def _write(root: Path, children: list[Path], **overrides: str) -> Path:
    return write_run_group(
        kind="replicates",
        group_id=overrides.get("group_id", "group_1"),
        name=overrides.get("name", "experiment"),
        config={"name": "experiment"},
        report={"trials": [{"run_dir": str(child)} for child in children]},
        report_json=None,
        models_dir=root,
    )


def _link(link: Path, target: Path) -> None:
    if sys.platform == "win32":
        import _winapi

        _winapi.CreateJunction(str(target), str(link))
    else:
        link.symlink_to(target, target_is_directory=True)


@pytest.mark.parametrize("field", ["name", "group_id"])
@pytest.mark.parametrize(
    "value",
    ["../escaped", "..\\escaped", "/absolute", "C:\\escaped", "part/name", ".", ".."],
)
def test_invalid_group_components_never_tag_children(
    tmp_path: Path, field: str, value: str
) -> None:
    """Invalid identifiers fail before the first otherwise valid child is tagged."""
    root = tmp_path / "models"
    child = _run(root / "child")
    before = (child / "run.json").read_bytes()
    with pytest.raises(ValueError):
        _write(root, [child], **{field: value})
    assert (child / "run.json").read_bytes() == before
    assert sorted(tmp_path.rglob("run.json")) == [child / "run.json"]


@pytest.mark.parametrize(
    "kind", ["absolute", "relative", "directory_alias", "json_alias", "temporary_alias"]
)
def test_external_child_never_changes_any_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """Every child and its staging path are checked before any metadata changes."""
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "models"
    valid = _run(root / "valid")
    outside = _run(tmp_path / "outside")
    child = outside
    if kind == "relative":
        child = Path("models/../outside")
    elif kind == "directory_alias":
        child = root / "alias"
        _link(child, outside)
    elif kind in {"json_alias", "temporary_alias"}:
        child = root / "linked_file"
        child.mkdir()
        filename = "run.json" if kind == "json_alias" else "run.json.tmp"
        # Hard links need no Windows symlink privilege and exercise shared file identity.
        (child / filename).hardlink_to(outside / "run.json")
    snapshots = {path: path.read_bytes() for path in tmp_path.rglob("run.json")}
    with pytest.raises(ValueError):
        _write(root, [valid, child])
    for path, before in snapshots.items():
        assert path.read_bytes() == before
    assert not (root / "experiment_replicates").exists()


@pytest.mark.parametrize("alias_location", ["group", "group_id"])
def test_group_alias_cannot_escape_before_child_tagging(
    tmp_path: Path, alias_location: str
) -> None:
    """Aliases in the group directory cannot redirect group JSON writes outside root."""
    root = tmp_path / "models"
    child = _run(root / "child")
    outside = _run(tmp_path / "outside")
    if alias_location == "group":
        _link(root / "experiment_replicates", outside)
    else:
        parent = root / "experiment_replicates"
        parent.mkdir()
        _link(parent / "group_1", outside)
    before = (child / "run.json").read_bytes()
    with pytest.raises(ValueError):
        _write(root, [child])
    assert (child / "run.json").read_bytes() == before
    assert json.loads((outside / "run.json").read_text()) == {"sentinel": True}


def test_explicit_external_model_root_remains_supported(tmp_path: Path) -> None:
    """The chosen root may be outside outputs and contain valid Unicode names."""
    root = tmp_path / "research-models"
    child = _run(root / "child")
    run_dir = _write(root, [child], name="Café trial 1")
    assert run_dir.is_relative_to(root)
    assert json.loads((child / "run.json").read_text())["group_id"] == "group_1"
    assert (
        json.loads((run_dir / "run.json").read_text())["experiment"] == "Café trial 1"
    )


def test_directory_builder_rejects_parent_components(tmp_path: Path) -> None:
    """The public path builder rejects traversal even without writing a group."""
    with pytest.raises(ValueError):
        group_run_dir(tmp_path, "../escaped", "replicates", "group_1")


def test_selected_root_alias_accepts_its_canonical_children(tmp_path: Path) -> None:
    """An explicitly selected root may itself be an alias to the user's storage."""
    actual = tmp_path / "storage"
    child = _run(actual / "child")
    selected = tmp_path / "selected-models"
    _link(selected, actual)
    run_dir = _write(selected, [child])
    assert run_dir.is_relative_to(actual)
    assert json.loads((child / "run.json").read_text())["group_id"] == "group_1"
