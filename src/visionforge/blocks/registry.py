from __future__ import annotations

from visionforge.blocks.base import ExperimentBlock


class BlockRegistry:
    """Auto-discovers all registered ExperimentBlock subclasses."""

    @classmethod
    def discover(cls) -> dict[str, type[ExperimentBlock]]:
        """Return a mapping of class name to class for all known concrete blocks.

        Returns:
            Dict mapping block name to its class. Empty if no blocks are registered.
        """
        return {
            sub.__name__: sub
            for sub in cls._all_subclasses(ExperimentBlock)
            if not getattr(sub, "__abstractmethods__", None)
        }

    @classmethod
    def _all_subclasses(cls, base: type) -> list[type]:
        result = []
        for sub in base.__subclasses__():
            result.append(sub)
            result.extend(cls._all_subclasses(sub))
        return result


__all__ = ["BlockRegistry"]
