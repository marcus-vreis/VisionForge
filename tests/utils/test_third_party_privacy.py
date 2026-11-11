"""Third-party telemetry opt-outs (SEC-6)."""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import tomllib
import types
from pathlib import Path

import pytest

from visionforge.utils import third_party_privacy as tpp

_ENV = tuple(tpp._ENV_OPT_OUTS)

# Distributions that open network connections on their own or on request.
_KNOWN_NETWORK = {
    "ultralytics", "huggingface-hub", "datasets", "mlflow", "wandb", "dvc",
    "roboflow", "kaggle", "sentry-sdk", "posthog", "segment-analytics-python",
    "requests", "httpx", "boto3", "google-cloud-storage", "comet-ml", "neptune",
}  # fmt: skip
_ALLOWED_TRANSPORT = {"requests", "httpx"}  # plumbing, no telemetry of their own


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


class _Events:
    def __init__(self) -> None:
        self.enabled = True


class _Settings(dict):  # type: ignore[type-arg]
    """Records writes: the global settings file must never be touched."""

    writes = 0

    def update(self, *a: object, **k: object) -> None:  # type: ignore[override]
        type(self).writes += 1
        super().update(*a, **k)


def _fake_ultralytics(monkeypatch: pytest.MonkeyPatch) -> tuple[_Events, _Settings]:
    events = _Events()
    settings = _Settings(sync=True)
    _Settings.writes = 0
    root = types.ModuleType("ultralytics")
    root.settings = settings  # type: ignore[attr-defined]
    utils = types.ModuleType("ultralytics.utils")
    events_mod = types.ModuleType("ultralytics.utils.events")
    events_mod.events = events  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ultralytics", root)
    monkeypatch.setitem(sys.modules, "ultralytics.utils", utils)
    monkeypatch.setitem(sys.modules, "ultralytics.utils.events", events_mod)
    return events, settings


def test_sets_env_and_disables_ultralytics_events(
    clean_env: pytest.MonkeyPatch,
) -> None:
    events, _settings = _fake_ultralytics(clean_env)
    tpp.apply_opt_outs()
    import os

    for name in _ENV:
        assert os.environ[name]
    assert events.enabled is False


def test_the_global_ultralytics_settings_are_not_written(
    clean_env: pytest.MonkeyPatch,
) -> None:
    # settings.update would persist sync=False in the user's ultralytics settings
    # file and silence every other project on the machine.
    _events, settings = _fake_ultralytics(clean_env)
    tpp.apply_opt_outs()

    assert _Settings.writes == 0
    assert settings["sync"] is True


def test_user_value_is_not_overwritten(clean_env: pytest.MonkeyPatch) -> None:
    _fake_ultralytics(clean_env)
    clean_env.setenv("HF_HUB_DISABLE_TELEMETRY", "0")
    tpp.apply_opt_outs()
    import os

    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "0"


def test_idempotent(clean_env: pytest.MonkeyPatch) -> None:
    events, _settings = _fake_ultralytics(clean_env)
    tpp.apply_opt_outs()
    tpp.apply_opt_outs()
    assert events.enabled is False


def test_without_ultralytics_does_not_raise(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setitem(sys.modules, "ultralytics", None)  # import raises ImportError
    tpp.apply_opt_outs()


def test_broken_ultralytics_events_do_not_raise(
    clean_env: pytest.MonkeyPatch,
) -> None:
    mod = types.ModuleType("ultralytics.utils.events")  # no `events` attribute
    clean_env.setitem(sys.modules, "ultralytics", types.ModuleType("ultralytics"))
    clean_env.setitem(sys.modules, "ultralytics.utils", types.ModuleType("u"))
    clean_env.setitem(sys.modules, "ultralytics.utils.events", mod)
    tpp.apply_opt_outs()


def test_the_real_ultralytics_events_are_switched_off(
    clean_env: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("ultralytics")
    from ultralytics.utils.events import events

    clean_env.setattr(events, "enabled", True)
    tpp.apply_opt_outs()

    assert events.enabled is False


def _dependency_names() -> set[str]:
    root = Path(__file__).resolve().parents[2]
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    specs = list(data["project"]["dependencies"])
    for extra in data["project"].get("optional-dependencies", {}).values():
        specs.extend(extra)
    names = set()
    for spec in specs:
        m = re.match(r"\s*([A-Za-z0-9_.-]+)", spec)
        if m:
            names.add(m.group(1).lower().replace("_", "-"))
    return names


def test_every_network_sdk_in_pyproject_has_an_entry() -> None:
    declared = _dependency_names() & (_KNOWN_NETWORK - _ALLOWED_TRANSPORT)
    missing = declared - set(tpp.NETWORK_SDKS)
    assert not missing, f"add to NETWORK_SDKS (and opt out if it reports): {missing}"


_SRC = Path(__file__).resolve().parents[2] / "src" / "visionforge"
_OPT_OUT = "apply_opt_outs"


def _imports_ultralytics(node: ast.AST) -> bool:
    if isinstance(node, ast.ImportFrom):
        return (node.module or "").split(".")[0] == "ultralytics"
    if isinstance(node, ast.Import):
        return any(a.name.split(".")[0] == "ultralytics" for a in node.names)
    return False


def _calls_opt_out(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == _OPT_OUT
    )


def _scope_nodes(scope: ast.AST) -> list[ast.AST]:
    """The nodes of one function (or module), without those of nested functions."""
    found: list[ast.AST] = []
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        found.append(node)
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
            stack.extend(ast.iter_child_nodes(node))
    return found


def _line(node: ast.AST) -> int:
    """Line of a statement or expression (``ast.AST`` itself declares none)."""
    return int(getattr(node, "lineno", 0))


def _unguarded_ultralytics_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    scopes = [tree, *(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
    )]  # fmt: skip
    offenders: list[str] = []
    for scope in scopes:
        nodes = _scope_nodes(scope)
        opt_outs = [_line(n) for n in nodes if _calls_opt_out(n)]
        offenders.extend(
            f"{path.relative_to(_SRC).as_posix()}:{_line(n)}"
            for n in nodes
            if _imports_ultralytics(n) and not any(at < _line(n) for at in opt_outs)
        )
    return offenders


def test_every_ultralytics_import_switches_its_events_off_first() -> None:
    # `visionforge gui` no longer calls apply_opt_outs() up front (#258), so each
    # place that imports ultralytics has to, before the import (ADR-180).
    offenders = [
        where
        for path in sorted(_SRC.rglob("*.py"))
        if path.name != "third_party_privacy.py"
        for where in _unguarded_ultralytics_imports(path)
    ]
    assert offenders == []


def test_the_opt_out_prints_nothing_on_stdout(tmp_path: Path) -> None:
    # On a machine without an ultralytics settings file, importing it prints
    # "Creating new Ultralytics Settings ..." on stdout. The opt-out runs before
    # commands whose stdout is JSON (`audit-data`, `bundle`), so that line broke
    # their output in a fresh environment (#258, the wheel smoke in CI).
    pytest.importorskip("ultralytics")
    env = {**os.environ, "YOLO_CONFIG_DIR": str(tmp_path), "PYTHONIOENCODING": "utf-8"}
    code = (
        "from visionforge.utils.third_party_privacy import apply_opt_outs;"
        "apply_opt_outs()"
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        check=True,
    )

    assert done.stdout == ""
    assert list(tmp_path.rglob("settings.json")), "ultralytics did not start fresh"
