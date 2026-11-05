"""The Colab connection (ADR-163, phase 1) against a fake CLI: nothing reaches Google."""

from __future__ import annotations

import io
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from visionforge.gui.api import colab_connection as cc
from visionforge.gui.api.colab_jobs import ColabJobs
from visionforge.gui.api.colab_routes import build_colab_router
from visionforge.gui.api.profiles import Profile

AUTH_URL = "https://accounts.google.com/o/oauth2/auth?response_type=code&state=xyz"
USAGE_IDLE = "Current balance: 381.93 compute units\nUsage rate: 0.00/hr\nActive assignments: 0\n"
USAGE_T4 = "Current balance: 381.93 compute units\nUsage rate: 1.15/hr\nActive assignments: 1\n"


@pytest.fixture(autouse=True)
def vf_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("VISIONFORGE_HOME", str(tmp_path))
    return tmp_path


def _write_token(slug: str) -> None:
    token = cc.token_path(slug)
    token.parent.mkdir(parents=True, exist_ok=True)
    token.write_text("{}", encoding="utf-8")


class FakeColabCli:
    """Stands in for `subprocess.run` of the venv and the CLI; records every call."""

    def __init__(
        self, usage: str = USAGE_T4, email: str = "pessoa@example.com"
    ) -> None:
        self.calls: list[tuple[list[str], dict[str, str] | None]] = []
        self.usage = usage
        self.email = email
        self.pip_fails = False

    def __call__(
        self, argv: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append((argv, kwargs.get("env")))
        if argv[1:3] == ["-m", "venv"]:
            python = cc._venv_bin("python")
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_text("", encoding="utf-8")
            return self._done("")
        if argv[1:4] == ["-m", "pip", "install"]:
            if self.pip_fails:
                return self._done("ERROR: no network", code=1)
            cc.cli_path().write_text("", encoding="utf-8")
            return self._done("")
        command = argv[-1]
        if command == "version":
            return self._done(f"Version: {cc.PINNED_VERSION}\n")
        if command == "whoami":
            return self._done(
                f"Auth provider: oauth2\nEmail:         {self.email}\nExpires in:    59m\n"
            )
        if command == "usage":
            return self._done(self.usage)
        raise AssertionError(f"unexpected CLI call {argv}")

    @staticmethod
    def _done(out: str, code: int = 0) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], code, stdout=out, stderr="")


class KeptStdin(io.StringIO):
    """A stdin whose text survives `close()`, so the test can read what was sent."""

    def close(self) -> None:
        self.sent = self.getvalue()

    def getvalue(self) -> str:
        return getattr(self, "sent", None) or super().getvalue()


class FakeLoginProcess:
    """`colab sessions` waiting for a code: prints the URL, writes the token on the right code."""

    def __init__(self, slug: str, accepted_code: str, prints_url: bool = True) -> None:
        self.slug = slug
        self.accepted_code = accepted_code
        lines = "\nTo authorize colab-cli, visit this URL in any browser:\n\n"
        if prints_url:
            lines += f"  {AUTH_URL}\n\nAfter approving, Google will display an authorization code.\n"
        self.stdout = io.StringIO(lines)
        self.stdin = KeptStdin()
        self.returncode: int | None = None
        self.killed = False
        self.pid = 4242

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            ok = self.stdin.getvalue() == self.accepted_code + "\n"
            if ok:
                _write_token(self.slug)
            self.returncode = 0 if ok else 1
        return self.returncode

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


class FakePopen:
    def __init__(
        self, accepted_code: str = "4/abc-DEF", prints_url: bool = True
    ) -> None:
        self.accepted_code = accepted_code
        self.prints_url = prints_url
        self.started: list[FakeLoginProcess] = []
        self.envs: list[dict[str, str]] = []

    def __call__(self, argv: list[str], **kwargs: Any) -> FakeLoginProcess:
        assert argv[-1] == "sessions", (
            "the login must use a command that allocates nothing"
        )
        env = kwargs["env"]
        self.envs.append(env)
        slug = Path(env["USERPROFILE"]).name
        proc = FakeLoginProcess(slug, self.accepted_code, self.prints_url)
        self.started.append(proc)
        return proc


def _installed(fake: FakeColabCli) -> cc.ColabCli:
    cli = cc.ColabCli(run=fake, python="base-python")
    cli._install()
    assert cli.installed_version() == cc.PINNED_VERSION
    return cli


def _client(cli: cc.ColabCli, logins: cc.LoginFlows) -> TestClient:
    def resolve(header: str | None) -> Profile:
        slug = header or "default"
        return Profile(slug=slug, name=slug, models_dir=Path("models"))

    app = FastAPI()
    app.include_router(
        build_colab_router(resolve, cli, logins, ColabJobs()), prefix="/api"
    )
    return TestClient(app)


# ── parsing what the CLI prints (formats read in google-colab-cli 0.7.5) ──────


def test_parse_usage_reads_balance_rate_and_assignments() -> None:
    usage = cc.parse_usage(USAGE_T4)
    assert usage == cc.Usage(balance=381.93, rate_per_hour=1.15, active_assignments=1)
    assert usage.hours_left == pytest.approx(381.93 / 1.15)


def test_hours_left_is_none_while_nothing_runs() -> None:
    assert cc.parse_usage(USAGE_IDLE).hours_left is None


def test_parse_usage_of_unknown_output_leaves_fields_empty() -> None:
    assert cc.parse_usage("[colab] something else") == cc.Usage(None, None, None)


def test_parse_email_ignores_the_unavailable_placeholder() -> None:
    assert cc.parse_email("Email:         a.b@example.com\n") == "a.b@example.com"
    assert (
        cc.parse_email(
            "Email:         <unavailable: token has no userinfo.email scope>\n"
        )
        is None
    )


# ── install ───────────────────────────────────────────────────────────────────


def test_install_creates_own_venv_and_pins_the_version(vf_home: Path) -> None:
    fake = FakeColabCli()
    _installed(fake)
    venv_call, pip_call = fake.calls[0][0], fake.calls[1][0]
    assert venv_call == [
        "base-python",
        "-m",
        "venv",
        str(vf_home / "colab-cli" / "venv"),
    ]
    assert pip_call[-1] == f"google-colab-cli=={cc.PINNED_VERSION}"
    assert pip_call[0] == str(cc._venv_bin("python")), (
        "pip must run in the managed venv"
    )


def test_install_failure_is_reported_and_nothing_counts_as_installed() -> None:
    fake = FakeColabCli()
    fake.pip_fails = True
    cli = cc.ColabCli(run=fake, python="base-python")
    cli._install()
    assert cli.install_state == "failed"
    assert cli.install_error is not None
    assert "no network" in cli.install_error
    assert cli.installed_version() is None


def test_start_install_refuses_a_second_concurrent_install() -> None:
    cli = cc.ColabCli(run=FakeColabCli(), python="base-python")
    cli._install_state = "installing"
    assert cli.start_install() is False


# ── the profile's own CLI home ────────────────────────────────────────────────


def test_every_profile_gets_its_own_cli_home(vf_home: Path) -> None:
    fake = FakeColabCli()
    cli = _installed(fake)
    _write_token("ana")
    _write_token("bia")
    cli.account("ana")
    cli.account("bia")
    homes = [
        env["USERPROFILE"] for argv, env in fake.calls if argv[-1] == "whoami" and env
    ]
    assert homes == [str(vf_home / "colab" / "ana"), str(vf_home / "colab" / "bia")]
    whoami_env = next(env for argv, env in fake.calls if argv[-1] == "whoami" and env)
    assert whoami_env["HOME"] == whoami_env["USERPROFILE"]
    assert whoami_env["PYTHONUNBUFFERED"] == "1"


def test_account_is_none_without_a_token_and_does_not_call_the_cli() -> None:
    fake = FakeColabCli()
    cli = _installed(fake)
    calls_before = len(fake.calls)
    assert cli.account("ana") is None
    assert len(fake.calls) == calls_before


def test_account_is_cached_until_the_token_changes() -> None:
    fake = FakeColabCli()
    cli = _installed(fake)
    _write_token("ana")
    assert cli.account("ana") == "pessoa@example.com"
    assert cli.account("ana") == "pessoa@example.com"
    assert sum(argv[-1] == "whoami" for argv, _ in fake.calls) == 1


def test_logout_removes_only_that_profiles_token() -> None:
    cli = _installed(FakeColabCli())
    _write_token("ana")
    _write_token("bia")
    cli.logout("ana")
    assert not cli.logged_in("ana")
    assert cli.logged_in("bia")


# ── login ─────────────────────────────────────────────────────────────────────


def test_login_returns_the_url_and_the_right_code_writes_the_token() -> None:
    popen = FakePopen(accepted_code="4/abc-DEF")
    logins = cc.LoginFlows(popen=popen)
    assert logins.start("ana") == AUTH_URL
    assert logins.pending("ana")
    logins.finish("ana", "  4/abc-DEF \n")
    assert cc.token_path("ana").is_file()
    assert not logins.pending("ana")


def test_a_rejected_code_raises_and_leaves_no_token() -> None:
    logins = cc.LoginFlows(popen=FakePopen(accepted_code="4/right"))
    logins.start("ana")
    with pytest.raises(cc.ColabError):
        logins.finish("ana", "4/wrong")
    assert not cc.token_path("ana").exists()


@pytest.mark.parametrize("code", ["", "   ", "4/a b", "x" * (cc.MAX_CODE_LENGTH + 1)])
def test_malformed_codes_never_reach_the_cli(code: str) -> None:
    popen = FakePopen()
    logins = cc.LoginFlows(popen=popen)
    logins.start("ana")
    with pytest.raises(cc.ColabError):
        logins.finish("ana", code)
    assert popen.started[0].stdin.getvalue() == ""
    assert logins.pending("ana"), "a typo must not throw away the waiting login"


def test_finish_without_start_raises() -> None:
    logins = cc.LoginFlows(popen=FakePopen())
    with pytest.raises(cc.ColabError):
        logins.finish("ana", "4/abc")


def test_no_url_kills_the_process(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cc, "LOGIN_URL_TIMEOUT_S", 0.5)
    popen = FakePopen(prints_url=False)
    logins = cc.LoginFlows(popen=popen)
    with pytest.raises(cc.ColabError):
        logins.start("ana")
    assert popen.started[0].killed


def test_starting_again_kills_the_previous_waiting_login() -> None:
    popen = FakePopen()
    logins = cc.LoginFlows(popen=popen)
    logins.start("ana")
    logins.start("ana")
    assert popen.started[0].killed
    assert not popen.started[1].killed


# ── routes ────────────────────────────────────────────────────────────────────


def test_status_before_install() -> None:
    client = _client(
        cc.ColabCli(run=FakeColabCli(), python="p"), cc.LoginFlows(popen=FakePopen())
    )
    body = client.get("/api/colab/status").json()
    assert body == {
        "pinned_version": cc.PINNED_VERSION,
        "installed_version": None,
        "install_state": "idle",
        "install_error": None,
        "logged_in": False,
        "account": None,
        "login_pending": False,
    }


def test_install_route_runs_in_the_background_until_installed() -> None:
    cli = cc.ColabCli(run=FakeColabCli(), python="p")
    client = _client(cli, cc.LoginFlows(popen=FakePopen()))
    assert client.post("/api/colab/install").status_code == 200
    deadline = time.monotonic() + 5
    while cli.install_state == "installing" and time.monotonic() < deadline:
        time.sleep(0.01)
    assert (
        client.get("/api/colab/status").json()["installed_version"] == cc.PINNED_VERSION
    )


def test_login_needs_the_cli_installed() -> None:
    client = _client(
        cc.ColabCli(run=FakeColabCli(), python="p"), cc.LoginFlows(popen=FakePopen())
    )
    assert client.post("/api/colab/login/start").status_code == 409


def test_login_round_trip_through_the_routes_per_profile() -> None:
    client = _client(_installed(FakeColabCli()), cc.LoginFlows(popen=FakePopen("4/ok")))
    headers = {"X-VF-Profile": "ana"}
    assert client.post("/api/colab/login/start", headers=headers).json() == {
        "url": AUTH_URL
    }
    body = client.post(
        "/api/colab/login/finish", json={"code": "4/ok"}, headers=headers
    ).json()
    assert body["logged_in"] is True
    assert body["account"] == "pessoa@example.com"
    other = client.get("/api/colab/status", headers={"X-VF-Profile": "bia"}).json()
    assert other["logged_in"] is False


def test_a_wrong_code_is_a_502_without_echoing_it() -> None:
    client = _client(_installed(FakeColabCli()), cc.LoginFlows(popen=FakePopen("4/ok")))
    client.post("/api/colab/login/start")
    response = client.post("/api/colab/login/finish", json={"code": "4/secret-wrong"})
    assert response.status_code == 502
    assert "4/secret-wrong" not in response.text


def test_logout_route_signs_the_profile_out() -> None:
    client = _client(_installed(FakeColabCli()), cc.LoginFlows(popen=FakePopen()))
    _write_token("default")
    assert client.post("/api/colab/logout").json()["logged_in"] is False


def test_usage_route_reports_hours_left() -> None:
    client = _client(
        _installed(FakeColabCli(usage=USAGE_T4)), cc.LoginFlows(popen=FakePopen())
    )
    assert client.get("/api/colab/usage").status_code == 409
    _write_token("default")
    body = client.get("/api/colab/usage").json()
    assert body["balance"] == 381.93
    assert body["rate_per_hour"] == 1.15
    assert body["active_assignments"] == 1
    assert body["hours_left"] == pytest.approx(332.1, abs=0.1)
