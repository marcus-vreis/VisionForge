from __future__ import annotations

import os
import subprocess
import sys

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
