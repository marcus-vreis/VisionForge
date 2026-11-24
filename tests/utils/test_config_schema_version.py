from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from visionforge.utils.config import (
    CURRENT_SCHEMA_VERSION,
    CrossValidationConfig,
    ExperimentConfig,
    load_config,
    migrate_config_dict,
)
from visionforge.utils.messages import using_lang


@pytest.fixture(autouse=True)
def _english_messages() -> Iterator[None]:
    """These tests check the validation rules against their English wording;
    the Portuguese default is covered by tests/utils/test_messages.py (ADR-116)."""
    with using_lang("en"):
        yield


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
        raw = _raw(tmp_path, {"schema_version": CURRENT_SCHEMA_VERSION + 1})
        with pytest.raises(ValidationError, match="newer version"):
            ExperimentConfig.model_validate(raw)

    def test_zero_or_negative_rejected(self, tmp_path: Path) -> None:
        raw = _raw(tmp_path, {"schema_version": 0})
        with pytest.raises(ValidationError):
            ExperimentConfig.model_validate(raw)


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


class TestCrossValidationSelectOn:
    """`cross_validation.select_on` (ADR-242): v3, and old K-folds stay "best"."""

    def test_the_default_is_best(self) -> None:
        assert CrossValidationConfig().select_on == "best"

    def test_last_is_accepted(self) -> None:
        assert CrossValidationConfig(select_on="last").select_on == "last"

    def test_another_rule_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CrossValidationConfig.model_validate({"select_on": "median"})

    def test_a_v2_config_migrates_to_v3(self) -> None:
        out = migrate_config_dict({"name": "x", "schema_version": 2})
        assert out["schema_version"] == 3

    def test_a_v2_kfold_without_the_field_loads_as_best(self, tmp_path: Path) -> None:
        raw = _raw(
            tmp_path,
            {
                "schema_version": 2,
                "block": "cross_validation",
                "cross_validation": {"n_folds": 3},
            },
        )
        cfg_path = tmp_path / "v2_kfold.yaml"
        cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

        cfg = load_config(cfg_path)

        assert cfg.cross_validation is not None
        assert cfg.cross_validation.select_on == "best"


class TestSelectOnLastWithEarlyStopping:
    """`select_on: last` with early stopping is refused (ADR-242 addendum).

    Early stopping reads the fold's own validation split to decide when to
    stop, so "the last epoch" would still be chosen by that split.
    """

    def test_last_with_patience_is_refused(self, tmp_path: Path) -> None:
        raw = _raw(
            tmp_path,
            {
                "block": "cross_validation",
                "cross_validation": {"n_folds": 3, "select_on": "last"},
                "training": {"early_stopping_patience": 3},
            },
        )
        with pytest.raises(ValidationError, match="early_stopping_patience"):
            ExperimentConfig.model_validate(raw)

    def test_last_without_patience_is_accepted(self, tmp_path: Path) -> None:
        raw = _raw(
            tmp_path,
            {
                "block": "cross_validation",
                "cross_validation": {"n_folds": 3, "select_on": "last"},
                "training": {"early_stopping_patience": 0},
            },
        )
        cfg = ExperimentConfig.model_validate(raw)
        assert cfg.cross_validation is not None
        assert cfg.cross_validation.select_on == "last"

    def test_best_with_patience_is_accepted(self, tmp_path: Path) -> None:
        raw = _raw(
            tmp_path,
            {
                "block": "cross_validation",
                "cross_validation": {"n_folds": 3, "select_on": "best"},
                "training": {"early_stopping_patience": 3},
            },
        )
        cfg = ExperimentConfig.model_validate(raw)
        assert cfg.training.early_stopping_patience == 3


class TestSelectOnLastWithPlateau:
    """`select_on: last` with the plateau scheduler is refused (ADR-242 addendum).

    ReduceLROnPlateau lowers the learning rate from the fold's own validation
    loss, so the weights of the last epoch would still depend on that split.
    """

    def test_last_with_plateau_is_refused(self, tmp_path: Path) -> None:
        raw = _raw(
            tmp_path,
            {
                "block": "cross_validation",
                "cross_validation": {"n_folds": 3, "select_on": "last"},
                "training": {
                    "early_stopping_patience": 0,
                    "scheduler": {"kind": "plateau"},
                },
            },
        )
        with pytest.raises(ValidationError, match="plateau"):
            ExperimentConfig.model_validate(raw)

    @pytest.mark.parametrize(
        ("rule", "kind"), [("last", "cosine"), ("last", "step"), ("best", "plateau")]
    )
    def test_other_combinations_are_accepted(
        self, tmp_path: Path, rule: str, kind: str
    ) -> None:
        raw = _raw(
            tmp_path,
            {
                "block": "cross_validation",
                "cross_validation": {"n_folds": 3, "select_on": rule},
                "training": {
                    "early_stopping_patience": 0,
                    "scheduler": {"kind": kind},
                },
            },
        )
        cfg = ExperimentConfig.model_validate(raw)
        assert cfg.training.scheduler.kind == kind
