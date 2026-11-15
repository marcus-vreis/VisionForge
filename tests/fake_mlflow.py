"""Named in-memory fake of the optional ``mlflow`` client (ADR-170).

CI does not install the ``mlflow`` extra. The ``fake_mlflow`` fixture in
``tests/conftest.py`` installs this module as ``mlflow`` so the export logic runs
through the same method names as the real ``MlflowClient``.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import pytest


@dataclass
class FakeMetric:
    key: str
    value: float
    timestamp: int
    step: int


@dataclass
class FakeRun:
    run_id: str
    experiment_id: str
    tags: dict[str, str]
    params: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, list[FakeMetric]] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    status: str = "RUNNING"

    @property
    def info(self) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            run_id=self.run_id, experiment_id=self.experiment_id
        )

    @property
    def data(self) -> types.SimpleNamespace:
        return types.SimpleNamespace(tags=self.tags)


class FakeMlflowClient:
    """In-memory stand-in for ``mlflow.MlflowClient`` shared by every instance."""

    experiments: dict[str, str] = {}
    runs: dict[str, FakeRun] = {}

    def __init__(self, tracking_uri: str) -> None:
        self.tracking_uri = tracking_uri

    @classmethod
    def reset(cls) -> None:
        cls.experiments = {}
        cls.runs = {}

    def get_experiment_by_name(self, name: str) -> types.SimpleNamespace | None:
        if name not in self.experiments:
            return None
        return types.SimpleNamespace(experiment_id=self.experiments[name])

    def create_experiment(self, name: str, artifact_location: str) -> str:
        self.experiments[name] = str(len(self.experiments) + 1)
        return self.experiments[name]

    def get_run(self, run_id: str) -> FakeRun:
        return self.runs[run_id]

    def search_runs(
        self, experiment_ids: list[str], filter_string: str
    ) -> list[FakeRun]:
        identity = filter_string.split("'")[1]
        return [
            run
            for run in self.runs.values()
            if run.experiment_id in experiment_ids
            and run.tags.get("vf.local_run") == identity
        ]

    def create_run(self, experiment_id: str, tags: dict[str, str]) -> FakeRun:
        run = FakeRun(f"run{len(self.runs) + 1}", experiment_id, dict(tags))
        self.runs[run.run_id] = run
        return run

    def log_param(self, run_id: str, key: str, value: str) -> None:
        self.runs[run_id].params[key] = value

    def get_metric_history(self, run_id: str, key: str) -> list[FakeMetric]:
        return list(self.runs[run_id].metrics.get(key, []))

    def log_batch(
        self, run_id: str, metrics: list[FakeMetric], synchronous: bool
    ) -> None:
        for metric in metrics:
            self.runs[run_id].metrics.setdefault(metric.key, []).append(metric)

    def log_artifact(self, run_id: str, local_path: str) -> None:
        self.runs[run_id].artifacts.append(Path(local_path).name)

    def set_terminated(self, run_id: str, status: str) -> None:
        self.runs[run_id].status = status


def install_fake_mlflow(monkeypatch: pytest.MonkeyPatch) -> type[FakeMlflowClient]:
    FakeMlflowClient.reset()
    module = types.ModuleType("mlflow")
    module.MlflowClient = FakeMlflowClient  # type: ignore[attr-defined]
    entities = types.ModuleType("mlflow.entities")
    entities.Metric = FakeMetric  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlflow", module)
    monkeypatch.setitem(sys.modules, "mlflow.entities", entities)
    return FakeMlflowClient
