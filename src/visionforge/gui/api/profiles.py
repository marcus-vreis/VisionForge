"""Profiles for a shared server: output folders, not accounts (ADR-114).

A profile IS a folder. ``outputs/profiles/<slug>/`` holds the usual output
sub-folders (``models``, ``graphics``, ``logs``, ``reports``) and a
``profile.json`` with the display name. Listing the profiles lists those
folders, creating one creates the folder; there is no user database.

The *default* profile is the folder layout that existed before profiles did
(``outputs/models`` and its siblings). It is what a request without the
``X-VF-Profile`` header resolves to, so nothing moves for an install that never
creates a profile, and runs started from the command line keep landing there.

Two things a profile changes, and only these two:

- where a submitted run writes (``Profile.scope_dict`` / ``Profile.scope_model``
  override ``config.output.*``), and
- the root the History reads (``Profile.models_dir``).

A profile is organisation, not access control: anyone who can reach the server
can send any profile's slug. What this module does enforce is that a slug can
never name a folder outside ``outputs/profiles`` (no separators, no dots, and
the resolved folder must be a direct child of the profiles root).
"""

from __future__ import annotations

import json
import re
import shutil
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from visionforge.utils.messages import tr

DEFAULT_SLUG = "default"
PROFILE_HEADER = "X-VF-Profile"
MAX_SLUG_LEN = 40
MAX_NAME_LEN = 60

# Sub-folders every profile has, named like the ``OutputConfig`` fields minus
# ``_dir``: ``models`` <-> ``models_dir``.
OUTPUT_SUBDIRS = ("models", "graphics", "logs", "reports")

# A slug starts with a letter or digit so it can never be read as an option
# ("-x") or hidden entry ("_x"); the rest is the alphabet the decision allows.
_SLUG_RE = re.compile(rf"[a-z0-9][a-z0-9_-]{{0,{MAX_SLUG_LEN - 1}}}")

# "default" is the legacy layout, and the DOS device names cannot be folder
# names on Windows (creating ``con`` there fails or opens a device). ``com0`` and
# ``lpt0`` count as devices too; the superscript forms (``com¹``) fold to
# ``com1`` in ``slugify`` before they ever reach this set.
_RESERVED = frozenset(
    {DEFAULT_SLUG, "con", "prn", "aux", "nul"}
    | {f"com{n}" for n in range(10)}
    | {f"lpt{n}" for n in range(10)}
)

_M = TypeVar("_M", bound=BaseModel)


class ProfileError(Exception):
    """Base of the errors the route layer turns into HTTP statuses."""


class InvalidProfileError(ProfileError):
    """The slug is not a safe profile name, or its folder escapes the root (400)."""


class UnknownProfileError(ProfileError):
    """The slug is well formed but no such profile folder exists (404)."""


class ProfileNameError(ProfileError):
    """The display name cannot become a profile (422)."""


class ProfileCreateError(ProfileError):
    """The OS refused to make the folders: permissions, full disk, read-only (500)."""


class ProfileExistsError(ProfileError):
    """A profile with that slug already exists (409)."""

    def __init__(self, slug: str, name: str) -> None:
        super().__init__(tr("profile.exists", name=name, slug=slug))
        self.slug = slug
        self.name = name


@dataclass(frozen=True)
class Profile:
    """The folders one request works in.

    ``root`` is None for the default profile, which keeps whatever output paths
    the config already carries; for a named profile it is
    ``outputs/profiles/<slug>`` and every output path is forced under it.
    ``models_dir`` is where the History reads for this profile.
    """

    slug: str
    name: str
    models_dir: Path
    root: Path | None = None

    @property
    def is_default(self) -> bool:
        return self.root is None

    def output_paths(self) -> dict[str, str] | None:
        """``output.*`` values for a named profile; None for the default."""
        if self.root is None:
            return None
        return {f"{sub}_dir": (self.root / sub).as_posix() for sub in OUTPUT_SUBDIRS}

    def scope_dict(self, config: dict[str, Any]) -> dict[str, Any]:
        """A copy of a config dict whose ``output`` points inside this profile.

        The default profile returns the dict untouched. Any ``output`` the
        browser sent is overridden: where a run writes is the profile's decision.
        """
        paths = self.output_paths()
        if paths is None:
            return config
        existing = config.get("output")
        scoped = dict(config)
        scoped["output"] = {**(existing if isinstance(existing, dict) else {}), **paths}
        return scoped

    def scope_model(self, config: _M) -> _M:
        """The same, for a validated config model (every task has ``.output``)."""
        paths = self.output_paths()
        if paths is None:
            return config
        output = config.output  # type: ignore[attr-defined]
        scoped = output.model_copy(update={k: Path(v) for k, v in paths.items()})
        return config.model_copy(update={"output": scoped})


def default_profile(models_dir: Path) -> Profile:
    """The profile that maps to the pre-profile layout (``outputs/models``)."""
    return Profile(slug=DEFAULT_SLUG, name=DEFAULT_SLUG, models_dir=models_dir)


def normalize_display_name(raw: str) -> str:
    """Collapse whitespace, drop control characters and cap the length."""
    printable = "".join(ch for ch in raw if ch.isprintable() or ch.isspace())
    return " ".join(printable.split())[:MAX_NAME_LEN]


def slugify(name: str) -> str:
    """Folder name for a display name; ``""`` when nothing usable is left.

    Accents fold to their base letter (``João`` -> ``joao``), every other
    non-ASCII character is dropped, runs of anything outside
    ``[a-z0-9_-]`` become one ``-``, and the ends are trimmed. The frontend
    mirrors this to preview the folder; the server is the authority.
    """
    folded = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9_-]+", "-", folded.lower()).strip("-_")
    return slug[:MAX_SLUG_LEN].rstrip("-_")


def is_valid_slug(slug: str) -> bool:
    """True for a well-formed, non-reserved slug."""
    return _SLUG_RE.fullmatch(slug) is not None and slug not in _RESERVED


def _profile_folder(slug: str, profiles_dir: Path) -> Path | None:
    """The profile's folder when it exists and really sits under the root.

    The slug regex already rules out separators and dots. This is the second
    lock: a link planted inside ``outputs/profiles`` that points elsewhere (or
    at another profile) resolves to something that is not ``<root>/<slug>``,
    and is refused rather than followed.
    """
    folder = profiles_dir / slug
    if not folder.is_dir():
        return None
    try:
        real = folder.resolve()
        root = profiles_dir.resolve()
    except OSError:
        raise InvalidProfileError(tr("profile.unresolvable", slug=slug)) from None
    if real.parent != root or real.name != slug:
        raise InvalidProfileError(tr("profile.outside_root", slug=slug))
    for sub in OUTPUT_SUBDIRS:
        # Every output folder, not only ``models``: a linked ``graphics`` or
        # ``logs`` would send a run's files out of the profile just the same.
        child = folder / sub
        if child.exists() and child.resolve().parent != real:
            raise InvalidProfileError(
                tr("profile.linked_subfolder", slug=slug, sub=sub)
            )
    return folder


def _read_name(folder: Path, slug: str) -> str:
    try:
        data = json.loads((folder / "profile.json").read_text(encoding="utf-8"))
        name = normalize_display_name(str(data["name"]))
    except (OSError, ValueError, KeyError, TypeError):
        return slug
    return name or slug


def _build(slug: str, folder: Path, profiles_dir: Path) -> Profile:
    relative_root = profiles_dir / slug
    return Profile(
        slug=slug,
        name=_read_name(folder, slug),
        models_dir=relative_root / "models",
        root=relative_root,
    )


def resolve_profile(
    header: str | None, profiles_dir: Path, default_models_dir: Path
) -> Profile:
    """The profile an ``X-VF-Profile`` header names (absent means default).

    Raises:
        InvalidProfileError: not a safe slug (``../x``, uppercase, too long...).
        UnknownProfileError: a safe slug with no folder under ``profiles_dir``.
    """
    slug = (header or "").strip()
    if not slug or slug == DEFAULT_SLUG:
        return default_profile(default_models_dir)
    if not is_valid_slug(slug):
        raise InvalidProfileError(
            tr("profile.invalid", slug=slug[:60], max_len=MAX_SLUG_LEN)
        )
    folder = _profile_folder(slug, profiles_dir)
    if folder is None:
        raise UnknownProfileError(tr("profile.unknown", slug=slug))
    return _build(slug, folder, profiles_dir)


def list_profiles(profiles_dir: Path, default_models_dir: Path) -> list[Profile]:
    """The default profile first, then every profile folder by display name."""
    found: list[Profile] = []
    if profiles_dir.is_dir():
        for entry in profiles_dir.iterdir():
            if not is_valid_slug(entry.name):
                continue
            try:
                folder = _profile_folder(entry.name, profiles_dir)
            except InvalidProfileError:
                continue
            if folder is not None:
                found.append(_build(entry.name, folder, profiles_dir))
    found.sort(key=lambda p: (p.name.casefold(), p.slug))
    return [default_profile(default_models_dir), *found]


def create_profile(name: str, profiles_dir: Path, default_models_dir: Path) -> Profile:
    """Create ``profiles_dir/<slug>`` with the output sub-folders.

    Raises:
        ProfileNameError: nothing usable is left of the name, or it is reserved.
        ProfileExistsError: that slug is already a profile.
        ProfileCreateError: the OS refused to make the folders (no permission,
            no space left, a file where the profiles folder should be). A
            half-made folder is removed so the same name can be tried again.
    """
    display = normalize_display_name(name)
    slug = slugify(display)
    if not slug:
        raise ProfileNameError(tr("profile.name_empty"))
    if slug in _RESERVED:
        raise ProfileNameError(tr("profile.name_reserved", slug=slug))

    folder = profiles_dir / slug
    made = False
    try:
        profiles_dir.mkdir(parents=True, exist_ok=True)
        try:
            folder.mkdir()
        except FileExistsError:
            # Only a folder that is there is "taken"; anything else that raised
            # FileExistsError (a file in the way) is an OS failure like the rest.
            if folder.is_dir():
                raise ProfileExistsError(slug, _read_name(folder, slug)) from None
            raise
        made = True
        for sub in OUTPUT_SUBDIRS:
            (folder / sub).mkdir()
        (folder / "profile.json").write_text(
            json.dumps(
                {
                    "name": display,
                    "slug": slug,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    except OSError as exc:
        if made:
            shutil.rmtree(folder, ignore_errors=True)
        reason = exc.strerror or type(exc).__name__
        raise ProfileCreateError(
            tr("profile.create_failed", slug=slug, reason=reason)
        ) from exc
    return _build(slug, folder, profiles_dir)


__all__ = [
    "DEFAULT_SLUG",
    "MAX_NAME_LEN",
    "MAX_SLUG_LEN",
    "OUTPUT_SUBDIRS",
    "PROFILE_HEADER",
    "InvalidProfileError",
    "Profile",
    "ProfileCreateError",
    "ProfileError",
    "ProfileExistsError",
    "ProfileNameError",
    "UnknownProfileError",
    "create_profile",
    "default_profile",
    "is_valid_slug",
    "list_profiles",
    "normalize_display_name",
    "resolve_profile",
    "slugify",
]
