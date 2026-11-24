"""``visionforge`` CLI dispatch: selftest, new-task, offline, no command, error exits.

Every heavy callee is patched where ``main`` imports it, so nothing trains,
installs ultralytics, opens a browser or starts a server.
"""

from __future__ import annotations

import contextvars
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

import visionforge.__main__ as cli
import visionforge.cli_strategies as cli_strategies
import visionforge.core.reproduce as reproduce
import visionforge.tasks.scaffold as scaffold
import visionforge.utils.logger as vf_logger
import visionforge.utils.selftest as selftest
import visionforge.utils.third_party_privacy as tpp
from visionforge.utils.selftest import SelfTestOutcome


@pytest.fixture(autouse=True)
def _quiet_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep ``main`` away from global state: env switches, ultralytics, log sinks."""
    monkeypatch.setattr(cli, "_use_utf8_console", lambda: None)
    monkeypatch.setattr(tpp, "apply_env_opt_outs", lambda: None)
    monkeypatch.setattr(tpp, "apply_opt_outs", lambda: None)
    monkeypatch.setattr(tpp, "offline_requested", lambda: False)
    monkeypatch.setattr(vf_logger, "setup_logger", lambda **kw: None)


def _invoke(monkeypatch: pytest.MonkeyPatch, args: list[str]) -> int:
    monkeypatch.setattr(sys, "argv", ["visionforge", *args])
    try:
        # A copy of the context keeps the language the command binds from leaking.
        contextvars.copy_context().run(cli.main)
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


class TestPeekTask:
    def test_reads_the_task_field(self, tmp_path: Path) -> None:
        path = tmp_path / "c.yaml"
        path.write_text(yaml.safe_dump({"task": "regression"}), encoding="utf-8")
        assert cli._peek_task(path) == "regression"

    def test_missing_task_defaults_to_multiclass(self, tmp_path: Path) -> None:
        path = tmp_path / "c.yaml"
        path.write_text(yaml.safe_dump({"name": "x"}), encoding="utf-8")
        assert cli._peek_task(path) == "multiclass"

    def test_a_yaml_that_is_not_a_mapping_defaults_to_multiclass(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "c.yaml"
        path.write_text("- a\n- b\n", encoding="utf-8")
        assert cli._peek_task(path) == "multiclass"


class TestNoCommandAndParsing:
    def test_no_command_prints_help_and_exits_zero(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert _invoke(monkeypatch, []) == 0
        out = capsys.readouterr().out
        assert "usage: visionforge" in out
        assert "selftest" in out

    def test_unknown_command_is_a_parse_error(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert _invoke(monkeypatch, ["frobnicate"]) == 2
        assert "invalid choice" in capsys.readouterr().err

    def test_run_without_a_config_is_a_parse_error(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert _invoke(monkeypatch, ["run"]) == 2
        assert "config" in capsys.readouterr().err

    def test_version_flag_prints_the_version(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert _invoke(monkeypatch, ["--version"]) == 0
        assert capsys.readouterr().out.startswith("visionforge ")


class TestOfflineFlag:
    def test_the_flag_enables_offline_mode_before_the_command_runs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        calls: list[str] = []

        def record_offline() -> None:
            calls.append("offline")

        def strategy_handled(path: Path, parser: Any) -> bool:
            calls.append("run")
            return True

        monkeypatch.setattr(tpp, "enable_offline_mode", record_offline)
        monkeypatch.setattr(cli, "_run_strategy_command", strategy_handled)

        code = _invoke(monkeypatch, ["run", str(tmp_path / "c.yaml"), "--offline"])

        assert code == 0
        assert calls == ["offline", "run"]

    def test_the_environment_request_enables_it_too(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        calls: list[str] = []

        def record_offline() -> None:
            calls.append("offline")

        def strategy_handled(path: Path, parser: Any) -> bool:
            return True

        monkeypatch.setattr(tpp, "offline_requested", lambda: True)
        monkeypatch.setattr(tpp, "enable_offline_mode", record_offline)
        monkeypatch.setattr(cli, "_run_strategy_command", strategy_handled)

        assert _invoke(monkeypatch, ["run", str(tmp_path / "c.yaml")]) == 0
        assert calls == ["offline"]


class TestStrategyEnvelopeErrors:
    def test_an_invalid_envelope_exits_2_with_the_reason(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        def bad_envelope(path: Path) -> None:
            raise ValueError("strategy kind 'zzz' is unknown")

        monkeypatch.setattr(cli_strategies, "read_envelope", bad_envelope)

        code = _invoke(monkeypatch, ["run", str(tmp_path / "c.yaml")])

        assert code == 2
        assert "visionforge: strategy kind 'zzz' is unknown" in capsys.readouterr().err


class TestReproduceLoggerFailure:
    def test_an_unwritable_log_dir_exits_2_in_one_line(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        def fake_plan(bundle: Path, **kwargs: Any) -> str:
            return "plan"

        def broken_logger(**kwargs: Any) -> None:
            raise OSError("cannot create\nlog dir")

        monkeypatch.setattr(reproduce, "plan_reproduction", fake_plan)
        monkeypatch.setattr(vf_logger, "setup_logger", broken_logger)

        code = _invoke(
            monkeypatch,
            ["reproduce", str(tmp_path / "b.zip"), "--output", str(tmp_path / "o")],
        )

        assert code == 2
        assert "visionforge: cannot create log dir\n" in capsys.readouterr().err


def _outcome(task: str, strategy: str, status: str) -> SelfTestOutcome:
    return SelfTestOutcome(task=task, strategy=strategy, status=status, detail="d")


class TestSelftestCommand:
    def _patch_run(
        self, monkeypatch: pytest.MonkeyPatch, outcomes: list[SelfTestOutcome]
    ) -> list[dict[str, Any]]:
        seen: list[dict[str, Any]] = []

        def fake_run(workdir: Path, **kwargs: Any) -> list[SelfTestOutcome]:
            seen.append({"workdir": workdir, **kwargs})
            return outcomes

        monkeypatch.setattr(selftest, "run_selftest", fake_run)
        return seen

    def test_quick_runs_only_the_simple_strategy_and_exits_zero(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        seen = self._patch_run(
            monkeypatch, [_outcome("regression", "simple", "passed")]
        )

        code = _invoke(
            monkeypatch,
            [
                "selftest",
                "--quick",
                "--tasks",
                "regression",
                "--workdir",
                str(tmp_path),
            ],
        )

        assert code == 0
        assert seen[0]["strategies"] == ("simple",)
        assert seen[0]["tasks"] == ("regression",)
        assert seen[0]["workdir"] == tmp_path
        assert "regression" in capsys.readouterr().out

    def test_a_failed_case_exits_1_and_the_json_lists_every_outcome(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patch_run(
            monkeypatch,
            [
                _outcome("regression", "simple", "passed"),
                _outcome("regression", "cv", "failed"),
            ],
        )
        target = tmp_path / "sub" / "out.json"

        code = _invoke(
            monkeypatch,
            [
                "selftest",
                "--tasks",
                "regression",
                "--strategies",
                "simple,cv",
                "--workdir",
                str(tmp_path),
                "--json",
                str(target),
            ],
        )

        assert code == 1
        rows = json.loads(target.read_text(encoding="utf-8"))
        assert [(r["strategy"], r["status"]) for r in rows] == [
            ("simple", "passed"),
            ("cv", "failed"),
        ]

    def test_all_selects_every_task_and_strategy(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        seen = self._patch_run(
            monkeypatch, [_outcome("classification", "simple", "passed")]
        )

        assert _invoke(monkeypatch, ["selftest", "--workdir", str(tmp_path)]) == 0
        assert seen[0]["tasks"] == selftest.TASKS
        assert seen[0]["strategies"] == selftest.STRATEGIES

    def test_an_unknown_task_or_strategy_exits_2_before_running(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        seen = self._patch_run(monkeypatch, [])

        code = _invoke(
            monkeypatch,
            [
                "selftest",
                "--tasks",
                "nope",
                "--strategies",
                "bogus",
                "--workdir",
                str(tmp_path),
            ],
        )

        assert code == 2
        assert seen == []

    def test_without_a_workdir_a_temp_folder_is_used(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = self._patch_run(
            monkeypatch, [_outcome("regression", "simple", "passed")]
        )

        code = _invoke(monkeypatch, ["selftest", "--quick", "--tasks", "regression"])

        assert code == 0
        assert seen[0]["workdir"].name.startswith("visionforge_selftest_")


class TestNewTaskCommand:
    def _patch_scaffold(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> list[tuple[str, bool, bool]]:
        received: list[tuple[str, bool, bool]] = []

        def fake_scaffold(
            key: str, *, package: bool = False, force: bool = False
        ) -> Path:
            received.append((key, package, force))
            return tmp_path / f"{key}.py"

        monkeypatch.setattr(scaffold, "scaffold_task", fake_scaffold)
        return received

    def test_passes_the_flags_through_and_exits_zero(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        received = self._patch_scaffold(monkeypatch, tmp_path)

        code = _invoke(monkeypatch, ["new-task", "my_task", "--package", "--force"])

        assert code == 0
        assert received == [("my_task", True, True)]

    def test_defaults_are_a_flat_file_without_force(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        received = self._patch_scaffold(monkeypatch, tmp_path)

        assert _invoke(monkeypatch, ["new-task", "abc"]) == 0
        assert received == [("abc", False, False)]

    @pytest.mark.parametrize(
        "error", [ValueError("bad key"), FileExistsError("exists")]
    )
    def test_a_refused_scaffold_exits_1(
        self, monkeypatch: pytest.MonkeyPatch, error: Exception
    ) -> None:
        def refuse(key: str, *, package: bool = False, force: bool = False) -> Path:
            raise error

        monkeypatch.setattr(scaffold, "scaffold_task", refuse)

        assert _invoke(monkeypatch, ["new-task", "Bad Key"]) == 1
