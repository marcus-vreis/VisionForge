"""In-memory stand-in for the ``comet_ml`` SDK, without network (ADR-200).

``FakeComet`` keeps every experiment by key, so a resumed experiment (same
key) sees what the earlier one logged; unlike W&B, Comet keeps a repeated step,
so the fake does too. ``end()`` sends the summary notification the real SDK
sends, with the OS user name in its title, unless the tracker replaced it.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import Any

import pytest

URL = "https://www.comet.example"
OS_USER = "someone-local"


class FakeExperimentConfig:
    """``comet_ml.ExperimentConfig``: keeps the keyword arguments it was built with."""

    def __init__(self, **kwargs: Any) -> None:
        self.values = kwargs


@dataclass
class FakeExperiment:
    key: str
    workspace: str
    project: str
    alive: bool = True
    flush_ok: bool = True
    parameters: dict[str, Any] = field(default_factory=dict)
    metrics: list[tuple[dict[str, float], int | None]] = field(default_factory=list)
    others: dict[str, Any] = field(default_factory=dict)
    notifications: list[str] = field(default_factory=list)
    ends: int = 0

    @property
    def url(self) -> str:
        return f"{URL}/{self.workspace}/{self.project}/{self.key}"

    def log_parameters(self, parameters: dict[str, Any]) -> None:
        self.parameters.update(parameters)

    def log_metrics(
        self, dic: dict[str, float], step: int | None = None, epoch: int | None = None
    ) -> None:
        self.metrics.append((dict(dic), step))

    def log_other(self, key: str, value: Any) -> None:
        self.others[key] = value

    def flush(self) -> bool:
        return self.flush_ok

    def send_notification(self, title: str, *args: Any, **kwargs: Any) -> None:
        self.notifications.append(title)

    def end(self) -> None:
        self.ends += 1
        self.send_notification(f"{OS_USER} Experiment summary", "finished", {})

    def steps(self) -> list[int | None]:
        return [step for _, step in self.metrics]


@dataclass
class FakeComet:
    """A Comet account: its default workspace, projects and experiments."""

    default_workspace: str = "ana"
    # (workspace, project) -> isPublic
    projects: dict[tuple[str, str], bool] = field(default_factory=dict)
    create_public: bool | None = None  # what the server reports after a create
    experiments: dict[str, FakeExperiment] = field(default_factory=dict)
    starts: list[dict[str, Any]] = field(default_factory=list)
    calls: list[tuple[str, Any]] = field(default_factory=list)
    fail_start: Exception | None = None
    start_dead: bool = False

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        module = types.ModuleType("comet_ml")
        module.API = self.api  # type: ignore[attr-defined]
        module.ExperimentConfig = FakeExperimentConfig  # type: ignore[attr-defined]
        module.start = self.start  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "comet_ml", module)

    def api(self, api_key: str | None = None, cache: bool = True) -> FakeComet:
        self.calls.append(("API", {"api_key": api_key, "cache": cache}))
        return self

    def get_default_workspace(self) -> str:
        self.calls.append(("get_default_workspace", None))
        return self.default_workspace

    def get_project(self, workspace: str, project_name: str) -> dict[str, Any] | None:
        self.calls.append(("get_project", (workspace, project_name)))
        if (workspace, project_name) not in self.projects:
            return None
        return {
            "projectName": project_name,
            "workspaceName": workspace,
            "isPublic": self.projects[(workspace, project_name)],
        }

    def create_project(
        self, workspace: str, project_name: str, public: bool = False
    ) -> dict[str, Any]:
        self.calls.append(("create_project", (workspace, project_name, public)))
        reported = public if self.create_public is None else self.create_public
        self.projects[(workspace, project_name)] = reported
        return {"projectId": "p1"}

    def start(self, **kwargs: Any) -> FakeExperiment:
        self.starts.append(kwargs)
        if self.fail_start is not None:
            raise self.fail_start
        key = kwargs["experiment_key"]
        if key not in self.experiments:
            self.experiments[key] = FakeExperiment(
                key, kwargs["workspace"], kwargs["project_name"]
            )
        experiment = self.experiments[key]
        experiment.alive = not self.start_dead
        return experiment


def install_fake_comet(monkeypatch: pytest.MonkeyPatch) -> FakeComet:
    """Use the fake for ``comet_ml``; the environment switches stay test-local."""
    from visionforge.core.comet_tracking import IMPORT_SWITCHES

    for name in IMPORT_SWITCHES:
        # setenv first so the teardown removes what the tracker sets.
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    sdk = FakeComet()
    sdk.install(monkeypatch)
    return sdk
