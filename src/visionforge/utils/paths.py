"""Portable directory components used by experiment configs and group writers."""

from typing import Annotated

from pydantic import AfterValidator, Field

from visionforge.utils.messages import tr

_INVALID_CHARACTERS = frozenset('/\\<>:"|?*')
_RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"} | {
    f"{prefix}{number}" for prefix in ("COM", "LPT") for number in "123456789¹²³"
}


def validate_path_component(value: str) -> str:
    """Reject names that cannot safely identify one directory on supported platforms."""
    stem = value.split(".", 1)[0].rstrip(" ").upper()
    if (
        not value.strip()
        or value in {".", ".."}
        or value.endswith((".", " "))
        or stem in _RESERVED_NAMES
        or any(
            char in _INVALID_CHARACTERS or ord(char) < 32 or ord(char) == 127
            for char in value
        )
    ):
        raise ValueError(tr("config.unsafe_path_component", value=value))
    return value


RunName = Annotated[
    str,
    Field(
        min_length=1,
        description="Nome de pasta sem separadores, caracteres reservados ou ponto/espaço no final.",
    ),
    AfterValidator(validate_path_component),
]
