"""Profile endpoints (ADR-114, ADR-126), split out of ``routes.py`` (ADR-132)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from loguru import logger

from visionforge.gui.api.profiles import (
    InvalidProfileError,
    Profile,
    ProfileCreateError,
    ProfileExistsError,
    ProfileNameError,
    UnknownProfileError,
    create_profile,
    list_profiles,
    update_profile,
)
from visionforge.gui.api.schemas import (
    ProfileCreateRequest,
    ProfileInfo,
    ProfileListResponse,
    ProfileUpdateRequest,
)

# (profiles_dir, models_dir), read on every request. routes.py owns both paths
# (tests and the CLI repoint them), so they are handed in, not imported back.
ProfileDirs = Callable[[], tuple[Path, Path]]

_BAD_REQUEST: dict[int | str, dict[str, Any]] = {
    400: {"description": "Invalid profile slug"}
}
_NOT_FOUND: dict[int | str, dict[str, Any]] = {
    404: {"description": "Profile not found"}
}
_CONFLICT: dict[int | str, dict[str, Any]] = {
    409: {"description": "Profile already exists"}
}
_UNPROCESSABLE: dict[int | str, dict[str, Any]] = {
    422: {
        "description": "Invalid request",
        # Keeps FastAPI's validation-error schema that a description-only entry would drop.
        "content": {
            "application/json": {
                "schema": {"$ref": "#/components/schemas/HTTPValidationError"}
            }
        },
    }
}
_SERVER_ERROR: dict[int | str, dict[str, Any]] = {
    500: {"description": "Profile could not be saved"}
}


def _profile_info(profile: Profile) -> ProfileInfo:
    return ProfileInfo(
        slug=profile.slug,
        name=profile.name,
        is_default=profile.is_default,
        pronoun=profile.pronoun,
    )


def build_profiles_router(dirs: ProfileDirs) -> APIRouter:
    """Router for /profiles; mount it under the ``/api`` router that sets the language."""
    router = APIRouter()

    @router.get("/profiles")
    async def get_profiles() -> ProfileListResponse:
        """Every profile, the default one first (ADR-114).

        Deliberately not behind ``current_profile``: the picker must be able to list
        them while the browser still holds a slug that no longer exists.
        """
        profiles_dir, models_dir = dirs()
        return ProfileListResponse(
            profiles=[_profile_info(p) for p in list_profiles(profiles_dir, models_dir)]
        )

    @router.post(
        "/profiles",
        status_code=201,
        responses={**_CONFLICT, **_UNPROCESSABLE, **_SERVER_ERROR},
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
        responses={**_BAD_REQUEST, **_NOT_FOUND, **_UNPROCESSABLE, **_SERVER_ERROR},
    )
    async def edit_profile(slug: str, req: ProfileUpdateRequest) -> ProfileInfo:
        """Change a profile's display name and/or pronoun, the default's too (ADR-126).

        Raises:
            HTTPException: 400 for an unsafe slug, 404 for one with no folder, 422
                when the new name is empty once tidied, 500 when the OS refuses to
                write the file.
        """
        profiles_dir, models_dir = dirs()
        try:
            profile = update_profile(
                slug, profiles_dir, models_dir, name=req.name, pronoun=req.pronoun
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
