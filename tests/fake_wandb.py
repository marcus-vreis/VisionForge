"""In-memory stand-ins for the ``wandb`` SDK and the W&B GraphQL API (ADR-172).

``FakeWandb`` keeps every run by id, so a resumed run (same id) sees what the
earlier one logged and drops steps it already has, as W&B does. ``FakeGraphQL``
answers the three queries ``wandb_tracking`` sends.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import Any

import pytest

from visionforge.core import wandb_tracking

BASE_URL = "https://wandb.example"


class FakeSettings:
    """``wandb.Settings``: keeps the keyword arguments it was built with."""

    model_fields = {"base_url": types.SimpleNamespace(default=BASE_URL)}

    def __init__(self, **kwargs: Any) -> None:
        self.values = kwargs


@dataclass
class FakeRun:
    run_id: str
    url: str = ""
    history: dict[int, dict[str, float]] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    exit_codes: list[int] = field(default_factory=list)

    def log(self, data: dict[str, float], step: int) -> None:
        if self.history and step <= max(self.history):
            return  # W&B drops a step that is not after the last one
        self.history[step] = dict(data)

    def finish(self, exit_code: int = 0) -> None:
        self.exit_codes.append(exit_code)


@dataclass
class FakeWandb:
    runs: dict[str, FakeRun] = field(default_factory=dict)
    inits: list[dict[str, Any]] = field(default_factory=list)
    fail_init: Exception | None = None

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        module = types.ModuleType("wandb")
        module.Settings = FakeSettings  # type: ignore[attr-defined]
        module.init = self.init  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "wandb", module)

    def init(self, **kwargs: Any) -> FakeRun:
        self.inits.append(kwargs)
        if self.fail_init is not None:
            raise self.fail_init
        run_id = kwargs["id"]
        if run_id not in self.runs:
            url = f"https://wandb.example/{kwargs['entity']}/{kwargs['project']}/runs/{run_id}"
            self.runs[run_id] = FakeRun(run_id, url)
        return self.runs[run_id]


@dataclass
class FakeGraphQL:
    """A W&B account: its entity and its projects' visibility."""

    entity: str = "ana"
    projects: dict[tuple[str, str], str] = field(default_factory=dict)
    create_access: str | None = None  # what the server answers on create
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def __call__(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((query, variables))
        if "viewer" in query:
            return {"viewer": {"entity": self.entity}}
        if "upsertModel" in query:
            data = variables["input"]
            access = self.create_access or data["access"]
            self.projects[(data["entityName"], data["name"])] = access
            return {"upsertModel": {"project": {"access": access}}}
        key = (variables["entity"], variables["name"])
        if key not in self.projects:
            return {"project": None}
        return {"project": {"access": self.projects[key]}}


def install_fake_wandb(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[FakeWandb, FakeGraphQL]:
    """Use the fakes for ``wandb`` and for every GraphQL call the tracker makes."""
    sdk = FakeWandb()
    sdk.install(monkeypatch)
    api = FakeGraphQL()
    monkeypatch.setattr(wandb_tracking, "http_graphql", lambda _base, _token: api)
    return sdk, api
