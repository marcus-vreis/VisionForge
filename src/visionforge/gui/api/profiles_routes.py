"""Profile endpoints (ADR-114, ADR-126), split out of ``routes.py`` (ADR-132)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from fastapi import APIRouter, HTTPException
from loguru import logger

from visionforge.gui.api.dataset_scope import current_scope
from visionforge.gui.api.http_responses import (
    TAG_PROFILES,
    UNPROCESSABLE_TEXT,
    VALIDATION_ERROR_CONTENT,
)
from visionforge.gui.api.profiles import (
    InvalidProfileError,
    Profile,
    ProfileCreateError,
    ProfileExistsError,
    ProfileNameError,
    UnknownProfileError,
    create_profile,
    list_profiles,
    resolve_profile,
    update_profile,
)
from visionforge.gui.api.schemas import (
    ProfileCreateRequest,
    ProfileInfo,
    ProfileListResponse,
    ProfileUpdateRequest,
)
from visionforge.utils.messages import tr

# (profiles_dir, models_dir), read on every request. routes.py owns both paths
# (tests and the CLI repoint them), so they are handed in, not imported back.
ProfileDirs = Callable[[], tuple[Path, Path]]

_SAVE_FAILED_TEXT = "Profile could not be saved"


def _profile_info(profile: Profile) -> ProfileInfo:
    return ProfileInfo(
        slug=profile.slug,
        name=profile.name,
        is_default=profile.is_default,
        pronoun=profile.pronoun,
        settings=profile.settings,
    )


def _signed_in() -> str | None:
    """The profile of the signed-in person on a server; None on a local install.

    There the profile is the account the login proxy named (ADR-213): a person
    sees and changes that one only (ADR-223).
    """
    scope = current_scope()
    return scope.owner if scope is not None else None


def _reject_for_server(slug: str, req: ProfileUpdateRequest, own: str | None) -> None:
    """404 for somebody else's profile, 422 for local MLflow, on a server."""
    if own is None:
        return
    if slug != own:
        raise HTTPException(404, tr("profile.unknown", slug=slug[:60]))
    mlflow = req.settings.mlflow if req.settings is not None else None
    if mlflow is not None and mlflow.enabled:
        raise HTTPException(422, tr("server.mlflow_closed"))


def build_profiles_router(dirs: ProfileDirs) -> APIRouter:
    """Router for /profiles; mount it under the ``/api`` router that sets the language."""
    router = APIRouter(tags=[TAG_PROFILES])

    @router.get("/profiles")
    async def get_profiles() -> ProfileListResponse:
        """Every profile, the default one first (ADR-114).

        Deliberately not behind ``current_profile``: the picker must be able to list
        them while the browser still holds a slug that no longer exists.
        """
        profiles_dir, models_dir = dirs()
        own = _signed_in()
        if own is not None:
            # A server lists one profile: the signed-in person's (ADR-223).
            mine = resolve_profile(own, profiles_dir, models_dir)
            return ProfileListResponse(profiles=[_profile_info(mine)])
        return ProfileListResponse(
            profiles=[_profile_info(p) for p in list_profiles(profiles_dir, models_dir)]
        )

    @router.post(
        "/profiles",
        status_code=201,
        responses={
            409: {"description": "Profile already exists"},
            422: {
                "description": UNPROCESSABLE_TEXT,
                "content": VALIDATION_ERROR_CONTENT,
            },
            500: {"description": _SAVE_FAILED_TEXT},
        },
    )
    async def add_profile(req: ProfileCreateRequest) -> ProfileInfo:
        """Create ``outputs/profiles/<slug>/`` for a display name (ADR-114).

        Raises:
            HTTPException: 422 when nothing usable is left of the name or it is
                reserved, 409 when that folder already exists, 500 when the OS
                refuses to make the folders (permissions, full disk).
        """
        profiles_dir, _ = dirs()
        try:
            profile = create_profile(req.name, profiles_dir, req.pronoun)
        except ProfileNameError as exc:
            raise HTTPException(422, str(exc)) from exc
        except ProfileExistsError as exc:
            raise HTTPException(409, str(exc)) from exc
        except ProfileCreateError as exc:
            logger.warning("GUI: could not create profile for {!r}: {}", req.name, exc)
            raise HTTPException(500, str(exc)) from exc
        logger.info("GUI: created profile {} at {}", profile.slug, profile.root)
        return _profile_info(profile)

    @router.put(
        "/profiles/{slug}",
        responses={
            400: {"description": "Invalid profile slug"},
            404: {"description": "Profile not found"},
            422: {
                "description": UNPROCESSABLE_TEXT,
                "content": VALIDATION_ERROR_CONTENT,
            },
            500: {"description": _SAVE_FAILED_TEXT},
        },
    )
    async def edit_profile(slug: str, req: ProfileUpdateRequest) -> ProfileInfo:
        """Change a profile's display name and/or pronoun, the default's too (ADR-126).

        Raises:
            HTTPException: 400 for an unsafe slug, 404 for one with no folder, 422
                when the new name is empty once tidied, 500 when the OS refuses to
                write the file. On a server, 404 for a profile that is not the
                signed-in person's and 422 for turning local MLflow on (ADR-223).
        """
        profiles_dir, models_dir = dirs()
        _reject_for_server(slug, req, _signed_in())
        try:
            profile = update_profile(
                slug,
                profiles_dir,
                models_dir,
                name=req.name,
                pronoun=req.pronoun,
                settings=req.settings,
            )
        except InvalidProfileError as exc:
            raise HTTPException(400, str(exc)) from exc
        except UnknownProfileError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ProfileNameError as exc:
            raise HTTPException(422, str(exc)) from exc
        except ProfileCreateError as exc:
            logger.warning("GUI: could not save profile {}: {}", slug, exc)
            raise HTTPException(500, str(exc)) from exc
        return _profile_info(profile)

    return router
