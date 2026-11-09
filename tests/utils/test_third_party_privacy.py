"""Third-party telemetry opt-outs (SEC-6)."""

from __future__ import annotations

import re
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


def _fake_ultralytics(
    monkeypatch: pytest.MonkeyPatch, sync: bool = True
) -> dict[str, object]:
    store: dict[str, object] = {"sync": sync}

    class Settings(dict):  # type: ignore[type-arg]
        def update(self, *a: object, **k: object) -> None:  # type: ignore[override]
            super().update(*a, **k)

    settings = Settings(store)
    mod = types.ModuleType("ultralytics")
    mod.settings = settings  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ultralytics", mod)
    return settings


def test_sets_env_and_disables_ultralytics_sync(
    clean_env: pytest.MonkeyPatch,
) -> None:
    settings = _fake_ultralytics(clean_env)
    tpp.apply_opt_outs()
    import os

    for name in _ENV:
        assert os.environ[name]
    assert settings["sync"] is False


def test_user_value_is_not_overwritten(clean_env: pytest.MonkeyPatch) -> None:
    _fake_ultralytics(clean_env)
    clean_env.setenv("HF_HUB_DISABLE_TELEMETRY", "0")
    tpp.apply_opt_outs()
    import os

    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "0"


def test_idempotent(clean_env: pytest.MonkeyPatch) -> None:
    settings = _fake_ultralytics(clean_env)
    tpp.apply_opt_outs()
    tpp.apply_opt_outs()
    assert settings["sync"] is False


def test_without_ultralytics_does_not_raise(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setitem(sys.modules, "ultralytics", None)  # import raises ImportError
    tpp.apply_opt_outs()


def test_broken_ultralytics_settings_do_not_raise(
    clean_env: pytest.MonkeyPatch,
) -> None:
    mod = types.ModuleType("ultralytics")
    mod.settings = object()  # type: ignore[attr-defined]  # no .get/.update
    clean_env.setitem(sys.modules, "ultralytics", mod)
    tpp.apply_opt_outs()


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
