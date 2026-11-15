"""Starting the GUI must not import the training stack (GUI-11, #202).

``import visionforge.gui.server`` used to take ~5 s because the route module
imported every block, runner and trainer, and with them torch, torchvision,
ultralytics and matplotlib. Each endpoint now imports what it runs.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from visionforge.utils.third_party_privacy import _ENV_OPT_OUTS

_SRC = Path(__file__).resolve().parents[2] / "src"
_HEAVY = ("torch", "torchvision", "ultralytics", "matplotlib")

# Run in a clean interpreter: the test process has long imported torch.
_PROBE = """
import json, os, sys
from fastapi import FastAPI
from visionforge.gui.server import app
assert isinstance(app, FastAPI)
app.openapi()  # builds every route's schema
print(json.dumps({
    "loaded": [m for m in sys.argv[1:] if m in sys.modules],
    "server_file": sys.modules["visionforge.gui.server"].__file__,
    "env": {k: os.environ.get(k) for k in %r},
}))
"""


@pytest.fixture(scope="module")
def probe() -> dict[str, Any]:
    env = {k: v for k, v in os.environ.items() if k not in _ENV_OPT_OUTS}
    # This checkout's src, not the editable install another checkout may own.
    env["PYTHONPATH"] = str(_SRC)
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    code = _PROBE % (tuple(_ENV_OPT_OUTS),)
    done = subprocess.run(  # noqa: S603 - fixed interpreter and script
        [sys.executable, "-c", code, *_HEAVY],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    result: dict[str, Any] = json.loads(done.stdout.strip().splitlines()[-1])
    return result


def test_probe_imports_this_checkout(probe: dict[str, Any]) -> None:
    assert Path(probe["server_file"]).resolve().is_relative_to(_SRC)


def test_creating_the_app_loads_no_training_stack(probe: dict[str, Any]) -> None:
    assert probe["loaded"] == []


def test_importing_the_server_still_switches_telemetry_off(
    probe: dict[str, Any],
) -> None:
    # The trainers used to set these on import (ADR-180); the server must now.
    assert probe["env"] == _ENV_OPT_OUTS


def test_core_public_names_resolve_lazily() -> None:
    from visionforge.core import DataModule, MetricsPlotter, Trainer
    from visionforge.core.data import DataModule as RealDataModule
    from visionforge.core.plotter import MetricsPlotter as RealPlotter
    from visionforge.core.trainer import Trainer as RealTrainer

    assert DataModule is RealDataModule
    assert MetricsPlotter is RealPlotter
    assert Trainer is RealTrainer


def test_blocks_public_names_resolve_lazily() -> None:
    from visionforge.blocks import ClassificationBlock
    from visionforge.blocks.classification import (
        ClassificationBlock as RealBlock,
    )

    assert ClassificationBlock is RealBlock


@pytest.mark.parametrize("package", ["visionforge.core", "visionforge.blocks"])
def test_lazy_packages_list_and_guard_their_names(package: str) -> None:
    module = importlib.import_module(package)

    assert set(module.__all__) <= set(dir(module))
    with pytest.raises(AttributeError, match="no_such_name"):
        getattr(module, "no_such_name")  # noqa: B009 - the lookup is the test


def test_custom_task_engine_resolves_on_first_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from visionforge.gui.api import custom_task_routes as custom_tasks_mod
    from visionforge.tasks.engine import GenericTaskEngine

    monkeypatch.setattr(custom_tasks_mod, "GenericTaskEngine", None)
    assert custom_tasks_mod._generic_task_engine() is GenericTaskEngine

    class _Fake:
        pass

    monkeypatch.setattr(custom_tasks_mod, "GenericTaskEngine", _Fake)
    assert custom_tasks_mod._generic_task_engine() is _Fake


@pytest.mark.parametrize(
    ("module_name", "function_name"),
    [
        ("visionforge.blocks.regression_cv", "run_regression_cross_validation"),
        ("visionforge.blocks.segmentation_cv", "run_segmentation_cross_validation"),
    ],
)
def test_cv_fronts_delegate_to_the_task_module(
    monkeypatch: pytest.MonkeyPatch, module_name: str, function_name: str
) -> None:
    from visionforge.gui.api import strategies

    calls: list[tuple[object, dict[str, Any]]] = []

    def fake(config: object, **kwargs: Any) -> str:
        calls.append((config, kwargs))
        return "report"

    monkeypatch.setattr(f"{module_name}.{function_name}", fake)
    config = object()

    assert getattr(strategies, function_name)(config, n_folds=3, seed=7) == "report"
    assert calls == [(config, {"n_folds": 3, "seed": 7})]
