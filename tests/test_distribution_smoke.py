"""A distribution smoke check must reject checkout imports and missing SPA files."""

from pathlib import Path

import pytest

from scripts.verify_distribution import (
    validate_package_origin,
    validate_run_metadata,
    validate_static_assets,
)


def test_origin_must_be_inside_environment_and_outside_checkout(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    prefix = tmp_path / "environment"
    package = prefix / "site-packages/visionforge/__init__.py"
    package.parent.mkdir(parents=True)
    package.write_text("")
    assert validate_package_origin(package, prefix, checkout) == package.parent
    source = checkout / "src/visionforge/__init__.py"
    source.parent.mkdir(parents=True)
    source.write_text("")
    with pytest.raises(ValueError, match="outside"):
        validate_package_origin(source, prefix, checkout)
    with pytest.raises(ValueError, match="checkout"):
        validate_package_origin(source, checkout, checkout)


def test_packaged_spa_requires_its_referenced_assets(tmp_path: Path) -> None:
    static = tmp_path / "gui/static"
    assets = static / "assets"
    assets.mkdir(parents=True)
    html = static / "index.html"
    html.write_text(
        '<script src="/assets/index.js"></script><link href="/assets/main.css">'
    )
    (assets / "index.js").write_text("export {}")
    with pytest.raises(FileNotFoundError):
        validate_static_assets(tmp_path)
    (assets / "main.css").write_text("body {}")
    assert validate_static_assets(tmp_path) == ["assets/index.js", "assets/main.css"]
    html.write_text('<script src="https://example.test/index.js"></script>')
    with pytest.raises(ValueError, match="no JavaScript"):
        validate_static_assets(tmp_path)


def test_asset_reference_cannot_escape_static_directory(tmp_path: Path) -> None:
    static = tmp_path / "gui/static"
    (static / "assets").mkdir(parents=True)
    (tmp_path / "outside.js").write_text("export {}")
    (static / "index.html").write_text(
        '<script src="/assets/../../../outside.js"></script>'
    )
    with pytest.raises(ValueError, match="Invalid installed asset"):
        validate_static_assets(tmp_path)


def test_run_schema_is_recorded_inside_config() -> None:
    report = {
        "device_used": "cpu",
        "history": [{}],
        "config": {"schema_version": 1},
        "environment": {"torch": "installed"},
    }
    validate_run_metadata(report)
    report["config"] = {}
    with pytest.raises(ValueError, match="schema version"):
        validate_run_metadata(report)
