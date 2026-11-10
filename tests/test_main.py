from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import visionforge.__main__ as cli
from visionforge.utils import third_party_privacy as tpp
from visionforge.utils.third_party_privacy import _ENV_OPT_OUTS

_SRC = Path(__file__).resolve().parents[1] / "src"

_EM_DASH_UTF8 = "—".encode()

# ultralytics reconfigures stdout when imported, which would hide the bug on a
# machine that has it; blocking it makes the run depend only on our own code.
_RUN_HELP = (
    "import runpy, sys\n"
    "sys.modules['ultralytics'] = None\n"
    "sys.argv = ['visionforge', '--help']\n"
    "runpy.run_module('visionforge.__main__', run_name='__main__')\n"
)


def test_help_is_utf8_even_on_a_legacy_console_codepage() -> None:
    env = {**os.environ, "PYTHONIOENCODING": "cp1252"}
    result = subprocess.run(
        [sys.executable, "-c", _RUN_HELP],
        capture_output=True,
        env=env,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert _EM_DASH_UTF8 in result.stdout
    assert b"\x97" not in result.stdout


# The `gui` command's own bootstrap, minus the listening server (#258).
_GUI_BOOTSTRAP = (
    "import json, os, sys\n"
    "import visionforge.gui.server as server\n"
    "server.start_server = lambda **_kw: None\n"
    "sys.argv = ['visionforge', 'gui', '--no-browser']\n"
    "from visionforge.__main__ import main\n"
    "main()\n"
    "print(json.dumps({\n"
    "    'loaded': [m for m in ('ultralytics', 'torch') if m in sys.modules],\n"
    "    'env': {k: os.environ.get(k) for k in %r},\n"
    "}))\n"
)


def test_the_gui_command_does_not_load_ultralytics(tmp_path: Path) -> None:
    # apply_opt_outs() imported ultralytics.utils.events (~1.7 s) to switch its
    # events off; the GUI imports ultralytics only when a detection run starts,
    # and that import switches them off itself (ADR-180).
    env = {k: v for k, v in os.environ.items() if k not in _ENV_OPT_OUTS}
    env["PYTHONPATH"] = str(_SRC)
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    result = subprocess.run(  # noqa: S603 - fixed interpreter and script
        [sys.executable, "-c", _GUI_BOOTSTRAP % (tuple(_ENV_OPT_OUTS),)],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        check=False,
        timeout=180,
    )

    assert result.returncode == 0, result.stderr[-2000:]
    probe = json.loads(result.stdout.strip().splitlines()[-1])
    assert probe["loaded"] == []
    assert probe["env"] == _ENV_OPT_OUTS


def test_other_commands_still_switch_ultralytics_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(cli, "_use_utf8_console", lambda: None)  # pytest's streams
    monkeypatch.setattr(tpp, "apply_env_opt_outs", lambda: calls.append("env"))
    monkeypatch.setattr(tpp, "apply_opt_outs", lambda: calls.append("full"))
    monkeypatch.setattr(
        sys, "argv", ["visionforge", "verify-bundle", str(tmp_path / "none.zip")]
    )

    with pytest.raises(SystemExit):
        cli.main()

    assert "full" in calls


def test_the_gui_command_sets_only_the_environment_switches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import visionforge.gui.server as server
    import visionforge.utils.logger as vf_logger

    calls: list[str] = []
    started: list[dict[str, object]] = []
    monkeypatch.setattr(cli, "_use_utf8_console", lambda: None)  # pytest's streams
    monkeypatch.setattr(vf_logger, "setup_logger", lambda: None)  # pytest's sinks
    monkeypatch.setattr(tpp, "apply_env_opt_outs", lambda: calls.append("env"))
    monkeypatch.setattr(tpp, "apply_opt_outs", lambda: calls.append("full"))
    monkeypatch.setattr(server, "start_server", lambda **kw: started.append(kw))
    monkeypatch.setattr(sys, "argv", ["visionforge", "gui", "--no-browser"])

    cli.main()

    assert calls == ["env"]
    assert len(started) == 1
