"""Colab connection endpoints (ADR-163, phase 1): install, login per profile, usage.

No ``from __future__ import annotations``: ``ProfileDep`` is built inside the
builder, and FastAPI resolves string annotations against module globals only.
"""

from collections.abc import Callable
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException

from visionforge.gui.api.colab_connection import (
    PINNED_VERSION,
    ColabCli,
    ColabError,
    LoginFlows,
)
from visionforge.gui.api.profiles import PROFILE_HEADER, Profile
from visionforge.gui.api.schemas import (
    ColabLoginFinishRequest,
    ColabLoginStart,
    ColabStatus,
    ColabUsage,
)
from visionforge.utils.messages import tr

_CONFLICT: dict[int | str, dict[str, Any]] = {
    409: {"description": "The CLI is not installed or the profile is not logged in"}
}
_BAD_GATEWAY: dict[int | str, dict[str, Any]] = {
    502: {"description": "The Colab CLI failed"}
}


def build_colab_router(
    resolve_profile: Callable[[str | None], Profile],
    cli: ColabCli,
    logins: LoginFlows,
) -> APIRouter:
    """Build the Colab endpoints. Handlers are sync: each one may wait on the CLI."""
    router = APIRouter()

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

    def require_installed() -> None:
        if cli.installed_version() != PINNED_VERSION:
            raise HTTPException(409, tr("colab.not_installed"))

    @router.get("/colab/status")
    def colab_status(profile: ProfileDep) -> ColabStatus:
        """Install state, login and account of the current profile."""
        return status(profile.slug)

    @router.post("/colab/install")
    def colab_install(profile: ProfileDep) -> ColabStatus:
        """Install the pinned CLI into VisionForge's own venv, in the background."""
        cli.start_install()
        return status(profile.slug)

    @router.post("/colab/login/start", responses={**_CONFLICT, **_BAD_GATEWAY})
    def colab_login_start(profile: ProfileDep) -> ColabLoginStart:
        """Start Google's login; nothing is allocated on Colab."""
        require_installed()
        try:
            return ColabLoginStart(url=logins.start(profile.slug))
        except ColabError as exc:
            raise HTTPException(502, str(exc)) from None

    @router.post("/colab/login/finish", responses={**_BAD_GATEWAY})
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

    @router.get("/colab/usage", responses={**_CONFLICT, **_BAD_GATEWAY})
    def colab_usage(profile: ProfileDep) -> ColabUsage:
        """Balance and rate of the profile's Google account."""
        require_installed()
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

    return router
