"""The four standalone tasks really trained in a job's child process (ADR-248).

``test_job_process`` trains a real classification in a child; the other tasks
were only checked with a stand-in block. These are ``slow``: each starts a
Python that loads the training stack and trains one epoch on the CPU, on the
sample dataset the GUI offers, with the configs ``visionforge selftest`` uses.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

import pytest

import visionforge
import visionforge.gui.api.execution as execution
import visionforge.gui.api.training_routes as training_routes
import visionforge.utils.selftest as selftest
from visionforge.gui.api.run_queue import QueuedJob
from visionforge.gui.api.sample_dataset import create_sample_dataset
from visionforge.utils.anomaly_config import AnomalyConfig
from visionforge.utils.detection_config import DetectionConfig
from visionforge.utils.regression_config import RegressionConfig
from visionforge.utils.segmentation_config import SegmentationConfig

SRC = Path(visionforge.__file__).resolve().parents[1]

Payload = Callable[[Path, Path], dict[str, Any]]
Execute = Callable[[Any, str], Coroutine[Any, Any, None]]

CASES: dict[str, tuple[Payload, type[Any], Execute]] = {
    "regression": (
        selftest._regression_payload,
        RegressionConfig,
        training_routes._execute_regression,
    ),
    "segmentation": (
        selftest._segmentation_payload,
        SegmentationConfig,
        training_routes._execute_segmentation,
    ),
    "anomaly": (
        selftest._anomaly_payload,
        AnomalyConfig,
        training_routes._execute_anomaly,
    ),
    "detection": (
        selftest._detection_payload,
        DetectionConfig,
        training_routes._execute_detection,
    ),
}


def _begin(run_id: str, task: str) -> execution.JobState:
    async def _nothing() -> None:
        return None

    job = QueuedJob(
        run_id=run_id,
        label=run_id,
        task=task,
        strategy="simple",
        start=_nothing,
        isolated=True,
    )
    execution._begin_job(job)
    return execution._job_state(run_id)


@pytest.mark.slow
@pytest.mark.parametrize("task", sorted(CASES))
def test_a_real_task_trains_in_a_child_on_the_cpu(
    task: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload, config_cls, execute = CASES[task]
    monkeypatch.setattr(execution, "_JOB_STATES", {})
    monkeypatch.setattr(execution, "_latest_run_id", None)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join([str(SRC), os.environ.get("PYTHONPATH", "")])
    )
    sample = create_sample_dataset(tmp_path / "datasets", task)
    config = config_cls.model_validate(payload(Path(sample.path), tmp_path / "work"))
    run_id = f"real_{task}"
    state = _begin(run_id, task)

    async def scenario() -> list[dict[str, Any]]:
        await asyncio.wait_for(execute(config, run_id), 600)
        return [event async for event in state.log.read()]

    events = asyncio.run(scenario())
    assert state.run["status"] == "completed", state.run["error"]
    epochs = [e["epoch"] for e in events if e.get("event") == "epoch_end"]
    assert epochs == [1]
    run_dir = Path(state.run["run_dir"])
    assert run_dir.is_relative_to(tmp_path / "work" / "models" / task)
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert record["status"] == "completed"
    assert record["metrics"]
