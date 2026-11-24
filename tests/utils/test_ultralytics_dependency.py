"""Ultralytics ships with the package (ADR-243): a damaged install must be clear."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path
from typing import Any

import pytest

from visionforge.utils import doctor
from visionforge.utils.messages import using_lang

_SRC = Path(__file__).resolve().parents[2] / "src"


def _project() -> dict[str, Any]:
    pyproject = tomllib.loads((_SRC.parent / "pyproject.toml").read_text("utf-8"))
    project: dict[str, Any] = pyproject["project"]
    return project


def _block_ultralytics(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every ``import ultralytics...`` raise ImportError, as a damaged install."""
    for name in [m for m in sys.modules if m.split(".")[0] == "ultralytics"]:
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setitem(sys.modules, "ultralytics", None)


def test_ultralytics_is_a_base_dependency() -> None:
    assert "ultralytics>=8.3" in _project()["dependencies"]


def test_detection_extra_stays_as_an_empty_alias() -> None:
    # A saved `[detection]` install line keeps resolving (ADR-106, ADR-243).
    assert _project()["optional-dependencies"]["detection"] == []


def test_missing_ultralytics_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    _block_ultralytics(monkeypatch)
    assert doctor.ultralytics_installed() is False


def test_installed_ultralytics_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor.importlib.util, "find_spec", lambda name: object())
    assert doctor.ultralytics_installed() is True


@pytest.mark.parametrize(
    "from_source, command",
    [
        (True, 'pip install -e "." --force-reinstall'),
        (False, 'pip install --force-reinstall "visionforge-studio"'),
    ],
)
def test_message_names_the_reinstall_line(
    monkeypatch: pytest.MonkeyPatch, from_source: bool, command: str
) -> None:
    monkeypatch.setattr(doctor, "installed_from_source", lambda: from_source)
    with using_lang("en"):
        message = doctor.ultralytics_missing_message()
    assert command in message


def test_message_offers_the_torchvision_backend() -> None:
    with using_lang("en"):
        message = doctor.ultralytics_missing_message()
    assert "torchvision" in message


def test_package_imports_and_reports_without_ultralytics(tmp_path: Path) -> None:
    """A fresh interpreter where ``import ultralytics`` fails, as a damaged install."""
    script = textwrap.dedent(
        """
        import sys
        sys.modules["ultralytics"] = None  # ImportError on any import of it
        from visionforge.core import detection_trainer
        from visionforge.gui import server  # noqa: F401  (routes, system info)
        from visionforge.utils import doctor, third_party_privacy
        third_party_privacy.apply_opt_outs()
        assert detection_trainer.YOLO is None
        assert doctor.ultralytics_installed() is False
        print("ok")
        """
    )
    environment = os.environ | {"PYTHONPATH": str(_SRC), "CUDA_VISIBLE_DEVICES": "-1"}
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip().endswith("ok")
