"""An anomaly history saved before ADR-241 is read under today's names."""

from __future__ import annotations

from typing import Any

from visionforge.core.anomaly_trainer import AnomalyEpochResult
from visionforge.core.history_keys import upgrade_anomaly_epoch, upgrade_run_history

_OLD_EPOCH = {
    "epoch": 1,
    "train_loss": 0.1,
    "val_auroc": 0.93,
    "val_image_f1": 0.8,
    "val_threshold": 0.4,
}


def _record(history: Any, **top: Any) -> dict[str, Any]:
    return {"config": {"task": "anomaly"}, "metrics": {}, "history": history, **top}


class TestUpgradeAnomalyEpoch:
    def test_the_old_auroc_is_read_as_the_test_auroc(self) -> None:
        assert upgrade_anomaly_epoch(_OLD_EPOCH) == {
            "epoch": 1,
            "train_loss": 0.1,
            "test_auroc": 0.93,
            "val_image_f1": 0.8,
            "val_threshold": 0.4,
        }

    def test_the_input_record_is_left_as_it_was(self) -> None:
        record = dict(_OLD_EPOCH)

        upgrade_anomaly_epoch(record)

        assert record == _OLD_EPOCH

    def test_a_current_record_is_unchanged(self) -> None:
        current = {"epoch": 2, "test_auroc": 0.9}

        assert upgrade_anomaly_epoch(current) == current

    def test_an_undefined_auroc_stays_undefined(self) -> None:
        assert upgrade_anomaly_epoch({"epoch": 1, "val_auroc": None}) == {
            "epoch": 1,
            "test_auroc": None,
        }

    def test_the_new_name_wins_when_both_are_present(self) -> None:
        both = {"epoch": 1, "val_auroc": 0.1, "test_auroc": 0.9}

        assert upgrade_anomaly_epoch(both) == {"epoch": 1, "test_auroc": 0.9}

    def test_a_resume_snapshot_from_before_the_rename_builds_an_epoch(self) -> None:
        result = AnomalyEpochResult(**upgrade_anomaly_epoch(_OLD_EPOCH))

        assert result.test_auroc == 0.93


class TestUpgradeRunHistory:
    def test_an_old_anomaly_run_is_read_with_test_auroc(self) -> None:
        upgraded = upgrade_run_history(_record([_OLD_EPOCH]))

        assert upgraded["history"] == [upgrade_anomaly_epoch(_OLD_EPOCH)]

    def test_the_saved_record_is_not_modified(self) -> None:
        record = _record([dict(_OLD_EPOCH)])

        upgrade_run_history(record)

        assert record["history"] == [_OLD_EPOCH]

    def test_rows_that_are_not_records_are_kept(self) -> None:
        upgraded = upgrade_run_history(_record(["row", _OLD_EPOCH]))

        assert upgraded["history"][0] == "row"

    def test_a_researchers_validation_auroc_keeps_its_name(self) -> None:
        """A custom task's ``val_auroc`` is a validation score (ADR-058)."""
        record = _record([_OLD_EPOCH], task="custom:defects")

        assert upgrade_run_history(record) is record

    def test_another_task_is_returned_as_it_is(self) -> None:
        record = {"config": {"task": "regression"}, "history": [{"val_r2": 0.5}]}

        assert upgrade_run_history(record) is record

    def test_a_record_without_a_history_list_is_returned_as_it_is(self) -> None:
        record = _record(None)

        assert upgrade_run_history(record) is record

    def test_a_record_whose_config_is_not_a_mapping_is_returned_as_it_is(
        self,
    ) -> None:
        record = {"config": "anomaly", "history": [_OLD_EPOCH]}

        assert upgrade_run_history(record) is record
