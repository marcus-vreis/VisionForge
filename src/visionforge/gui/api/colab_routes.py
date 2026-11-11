"""Colab connection endpoints (ADR-163, phase 1): install, login per profile, usage.

No ``from __future__ import annotations``: ``ProfileDep`` is built inside the
builder, and FastAPI resolves string annotations against module globals only.
"""

from collections.abc import Callable
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException

from visionforge.gui.api.colab_connection import (
    PINNED_VERSION,
    ColabCli,
    ColabError,
    LoginFlows,
)
from visionforge.gui.api.colab_jobs import ColabJobs
from visionforge.gui.api.http_responses import TAG_COLAB
from visionforge.gui.api.profiles import PROFILE_HEADER, Profile
from visionforge.gui.api.schemas import (
    ColabLoginFinishRequest,
    ColabLoginStart,
    ColabSessionInfo,
    ColabStatus,
    ColabStopSessionRequest,
    ColabUsage,
)
from visionforge.utils.messages import tr

_NOT_READY = "The CLI is not installed or the profile is not logged in"
_CLI_FAILED = "The Colab CLI failed"


def build_colab_router(
    resolve_profile: Callable[[str | None], Profile],
    cli: ColabCli,
    logins: LoginFlows,
    jobs: ColabJobs,
) -> APIRouter:
    """Build the Colab endpoints. Handlers are sync: each one may wait on the CLI."""
    router = APIRouter(tags=[TAG_COLAB])

    def profile_dep(
        x_vf_profile: Annotated[str | None, Header(alias=PROFILE_HEADER)] = None,
    ) -> Profile:
        return resolve_profile(x_vf_profile)

    ProfileDep = Annotated[Profile, Depends(profile_dep)]

    def status(slug: str) -> ColabStatus:
        installed = cli.installed_version()
        return ColabStatus(
            pinned_version=PINNED_VERSION,
            installed_version=installed,
            install_state=cli.install_state,
            install_error=cli.install_error,
            logged_in=cli.logged_in(slug),
            account=cli.account(slug) if installed else None,
            login_pending=logins.pending(slug),
        )

    def installed() -> bool:
        return cli.installed_version() == PINNED_VERSION

    @router.get("/colab/status")
    def colab_status(profile: ProfileDep) -> ColabStatus:
        """Install state, login and account of the current profile."""
        return status(profile.slug)

    @router.post("/colab/install")
    def colab_install(profile: ProfileDep) -> ColabStatus:
        """Install the pinned CLI into VisionForge's own venv, in the background."""
        cli.start_install()
        return status(profile.slug)

    @router.post(
        "/colab/login/start",
        responses={409: {"description": _NOT_READY}, 502: {"description": _CLI_FAILED}},
    )
    def colab_login_start(profile: ProfileDep) -> ColabLoginStart:
        """Start Google's login; nothing is allocated on Colab."""
        if not installed():
            raise HTTPException(409, tr("colab.not_installed"))
        try:
            return ColabLoginStart(url=logins.start(profile.slug))
        except ColabError as exc:
            raise HTTPException(502, str(exc)) from None

    @router.post("/colab/login/finish", responses={502: {"description": _CLI_FAILED}})
    def colab_login_finish(
        req: ColabLoginFinishRequest, profile: ProfileDep
    ) -> ColabStatus:
        """Hand the code Google showed to the waiting CLI."""
        try:
            logins.finish(profile.slug, req.code)
        except ColabError as exc:
            # `from None`: the code must not travel in a chained traceback.
            raise HTTPException(502, str(exc)) from None
        return status(profile.slug)

    @router.post("/colab/login/cancel")
    def colab_login_cancel(profile: ProfileDep) -> ColabStatus:
        logins.cancel(profile.slug)
        return status(profile.slug)

    @router.post("/colab/logout")
    def colab_logout(profile: ProfileDep) -> ColabStatus:
        """Forget this profile's login, to sign in with another account."""
        logins.cancel(profile.slug)
        cli.logout(profile.slug)
        return status(profile.slug)

    @router.get(
        "/colab/usage",
        responses={409: {"description": _NOT_READY}, 502: {"description": _CLI_FAILED}},
    )
    def colab_usage(profile: ProfileDep) -> ColabUsage:
        """Balance and rate of the profile's Google account."""
        if not installed():
            raise HTTPException(409, tr("colab.not_installed"))
        if not cli.logged_in(profile.slug):
            raise HTTPException(409, tr("colab.not_logged_in"))
        try:
            usage = cli.usage(profile.slug)
        except ColabError as exc:
            raise HTTPException(502, str(exc)) from None
        return ColabUsage(
            balance=usage.balance,
            rate_per_hour=usage.rate_per_hour,
            active_assignments=usage.active_assignments,
            hours_left=usage.hours_left,
        )

    @router.get("/colab/sessions", responses={409: {"description": _NOT_READY}})
    def colab_sessions(profile: ProfileDep) -> list[ColabSessionInfo]:
        """Runtimes alive on the account, so a job cut short is not left running."""
        if not installed() or not cli.logged_in(profile.slug):
            raise HTTPException(409, tr("colab.not_logged_in"))
        return [
            ColabSessionInfo(name=s.name, hardware=s.hardware, ours=s.ours)
            for s in jobs.sessions(profile.slug)
        ]

    @router.post(
        "/colab/sessions/stop",
        responses={409: {"description": _NOT_READY}, 502: {"description": _CLI_FAILED}},
    )
    def colab_stop_session(
        req: ColabStopSessionRequest, profile: ProfileDep
    ) -> list[ColabSessionInfo]:
        """Release one runtime by name and return what is still alive."""
        if not installed() or not cli.logged_in(profile.slug):
            raise HTTPException(409, tr("colab.not_logged_in"))
        if not jobs.stop_session(profile.slug, req.name):
            raise HTTPException(
                502, tr("colab.cli_failed", command="stop", detail=req.name)
            )
        return [
            ColabSessionInfo(name=s.name, hardware=s.hardware, ours=s.ours)
            for s in jobs.sessions(profile.slug)
        ]

    return router
