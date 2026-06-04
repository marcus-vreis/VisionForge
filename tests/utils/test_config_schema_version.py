from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from visionforge.utils.config import (
    CURRENT_SCHEMA_VERSION,
    ExperimentConfig,
    load_config,
    migrate_config_dict,
)


def _raw(tmp_path: Path, overrides: dict | None = None) -> dict:
    raw: dict = {
        "name": "exp",
        "task": "multiclass",
        "model": {"name": "resnet18", "num_classes": 2},
        "data": {"base_dir": str(tmp_path)},
    }
    if overrides:
        raw.update(overrides)
    return raw


class TestSchemaVersionField:
    def test_defaults_to_current_version(self, tmp_path: Path) -> None:
        cfg = ExperimentConfig.model_validate(_raw(tmp_path))
        assert cfg.schema_version == CURRENT_SCHEMA_VERSION

    def test_round_trips_in_model_dump(self, tmp_path: Path) -> None:
        cfg = ExperimentConfig.model_validate(_raw(tmp_path))
        dumped = cfg.model_dump(mode="json")
        assert dumped["schema_version"] == CURRENT_SCHEMA_VERSION

    def test_future_version_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match="newer version"):
            ExperimentConfig.model_validate(
                _raw(tmp_path, {"schema_version": CURRENT_SCHEMA_VERSION + 1})
            )

    def test_zero_or_negative_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError):
            ExperimentConfig.model_validate(_raw(tmp_path, {"schema_version": 0}))


class TestMigrateConfigDict:
    def test_adds_version_when_missing(self) -> None:
        out = migrate_config_dict({"name": "x"})
        assert out["schema_version"] == CURRENT_SCHEMA_VERSION

    def test_leaves_current_version_unchanged(self) -> None:
        src = {"name": "x", "schema_version": CURRENT_SCHEMA_VERSION}
        out = migrate_config_dict(src)
        assert out["schema_version"] == CURRENT_SCHEMA_VERSION

    def test_does_not_mutate_input(self) -> None:
        src = {"name": "x"}
        migrate_config_dict(src)
        assert "schema_version" not in src

    def test_non_dict_passes_through(self) -> None:
        assert migrate_config_dict([]) == []  # type: ignore[arg-type]


class TestLoadConfigMigration:
    def test_legacy_yaml_without_version_loads(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "legacy.yaml"
        cfg_path.write_text(yaml.safe_dump(_raw(tmp_path)), encoding="utf-8")
        cfg = load_config(cfg_path)
        assert cfg.schema_version == CURRENT_SCHEMA_VERSION

    def test_explicit_current_version_loads(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(
            yaml.safe_dump(_raw(tmp_path, {"schema_version": CURRENT_SCHEMA_VERSION})),
            encoding="utf-8",
        )
        cfg = load_config(cfg_path)
        assert cfg.schema_version == CURRENT_SCHEMA_VERSION

    def test_future_version_yaml_rejected(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "future.yaml"
        cfg_path.write_text(
            yaml.safe_dump(_raw(tmp_path, {"schema_version": 999})),
            encoding="utf-8",
        )
        with pytest.raises(ValidationError, match="newer version"):
            load_config(cfg_path)
