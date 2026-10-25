"""Execute the published-package smoke workflow with local command boundaries."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

_WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/install-smoke.yml"
_SPEC = "visionforge-studio==0.18.1"
_EXPRESSION = "${{ steps.pick.outputs.spec }}"
_LOG_COMMAND = """\
{ printf '%s' "${0##*/}"; printf '\\t%s' "$@"; printf '\\n'; } >> "$STUB_LOG"
"""


def _bash() -> str:
    if os.name == "nt":
        # Windows' bash.exe starts WSL; the workflow uses Git Bash instead.
        git = shutil.which("git")
        assert git is not None, "Git Bash is required to execute workflow steps"
        # Git hooks can prepend mingw64/libexec/git-core rather than cmd/.
        bash_path = next(
            (
                root / "bin/bash.exe"
                for root in Path(git).resolve().parents
                if (root / "bin/bash.exe").is_file()
            ),
            None,
        )
        assert bash_path is not None, f"Git Bash not found above: {git}"
        return str(bash_path)
    bash_command = shutil.which("bash")
    assert bash_command is not None, "Bash is required to execute workflow steps"
    return bash_command


def _run_workflow(
    tmp_path: Path,
    *,
    succeed_after: int,
    stale_index: bool = False,
    spec: str = _SPEC,
) -> tuple[subprocess.CompletedProcess[str], list[list[str]]]:
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    stubs = {
        "python": """\
if [ "$*" = '-m venv .venv' ]; then
  mkdir -p .venv/bin
  : > .venv/bin/activate
elif [ "$*" != '-m pip install --upgrade pip' ]; then
  exit 99
fi
""",
        "pip": """\
count=0
if [ -f "$STUB_COUNT" ]; then read -r count < "$STUB_COUNT"; fi
count=$((count + 1))
printf '%s\\n' "$count" > "$STUB_COUNT"
if [ "$count" -lt "$STUB_SUCCEED_AFTER" ]; then exit 23; fi
if [ "$STUB_STALE_INDEX" = 1 ]; then
  fresh=0
  for argument in "$@"; do
    if [ "$argument" = '--no-cache-dir' ]; then fresh=1; fi
  done
  if [ "$fresh" = 0 ]; then exit 23; fi
fi
""",
        "sleep": "exit 0\n",
        "visionforge": "exit 0\n",
    }
    for name, body in stubs.items():
        executable = stub_dir / name
        executable.write_text("#!/bin/bash\n" + _LOG_COMMAND + body, encoding="utf-8")
        executable.chmod(0o755)

    log = tmp_path / "commands.log"
    startup = tmp_path / "bash-env.sh"
    # Git Bash prepends its own /usr/bin to the inherited Windows PATH.
    startup.write_text('export PATH="$(cd "$STUB_BIN" && pwd):$PATH"\n')
    environment = os.environ.copy()
    environment.update(
        PATH=str(stub_dir) + os.pathsep + environment["PATH"],
        BASH_ENV=startup.as_posix(),
        STUB_BIN=stub_dir.as_posix(),
        STUB_LOG=log.as_posix(),
        STUB_COUNT=(tmp_path / "attempts").as_posix(),
        STUB_SUCCEED_AFTER=str(succeed_after),
        STUB_STALE_INDEX=str(int(stale_index)),
    )
    workflow = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["install"]["steps"]
    install_index = next(
        index
        for index, step in enumerate(steps)
        if step.get("name") == "pip install, as the guide says"
    )
    for index, step in enumerate(steps[install_index:]):
        step_environment = environment | {
            name: value.replace(_EXPRESSION, spec)
            for name, value in step.get("env", {}).items()
        }
        script = tmp_path / f"step-{index}.sh"
        script.write_text(step["run"].replace(_EXPRESSION, spec), encoding="utf-8")
        result = subprocess.run(
            [_bash(), "--noprofile", "--norc", "-e", "-o", "pipefail", str(script)],
            cwd=tmp_path,
            env=step_environment,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode:
            break
    calls = [line.split("\t") for line in log.read_text(encoding="utf-8").splitlines()]
    return result, calls


@pytest.mark.parametrize("succeed_after", [1, 3, 5])
def test_smoke_runs_cli_only_after_install(tmp_path: Path, succeed_after: int) -> None:
    """Successful installs retain the requested version across bounded retries."""
    result, calls = _run_workflow(tmp_path, succeed_after=succeed_after)

    assert result.returncode == 0, result.stderr
    installs = [call for call in calls if call[0] == "pip"]
    assert len(installs) == succeed_after
    assert all(call[-1] == _SPEC for call in installs)
    assert [call for call in calls if call[0] == "sleep"] == [["sleep", "60"]] * (
        succeed_after - 1
    )
    assert calls[-3:] == [
        ["visionforge", "--version"],
        ["visionforge", "doctor"],
        ["visionforge", "selftest", "--quick"],
    ]


def test_smoke_refreshes_stale_index_after_publication(tmp_path: Path) -> None:
    """A retry can see a published version after a stale initial index response."""
    result, calls = _run_workflow(tmp_path, succeed_after=2, stale_index=True)

    assert result.returncode == 0, result.stderr
    assert len([call for call in calls if call[0] == "pip"]) == 2
    assert [call for call in calls if call[0] == "sleep"] == [["sleep", "60"]]
    assert calls[-1] == ["visionforge", "selftest", "--quick"]


def test_smoke_preserves_install_failure_without_final_sleep(tmp_path: Path) -> None:
    """Exhaustion retains pip's error even when an older CLI is available."""
    result, calls = _run_workflow(tmp_path, succeed_after=6)

    assert result.returncode == 23, result.stderr
    installs = [call for call in calls if call[0] == "pip"]
    assert len(installs) == 5
    assert all(call[-1] == _SPEC for call in installs)
    assert [call for call in calls if call[0] == "sleep"] == [["sleep", "60"]] * 4
    assert not any(call[0] == "visionforge" for call in calls)


def test_smoke_passes_spec_as_one_literal_argument(tmp_path: Path) -> None:
    """The selected spec reaches pip unchanged through the workflow environment."""
    spec = "visionforge-studio[gui] == 0.18.1; $(touch injected)"
    result, calls = _run_workflow(tmp_path, succeed_after=1, spec=spec)

    assert result.returncode == 0, result.stderr
    install = next(call for call in calls if call[0] == "pip")
    assert install[-1] == spec
    assert not (tmp_path / "injected").exists()
