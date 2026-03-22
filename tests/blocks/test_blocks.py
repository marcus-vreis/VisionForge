from typing import Any

import pytest

from visionforge.blocks.base import ExperimentBlock
from visionforge.blocks.registry import BlockRegistry
from visionforge.utils.config import ExperimentConfig


class TestExperimentBlock:
    def test_cannot_instantiate_abstract(self) -> None:
        """ExperimentBlock must not be directly instantiable."""
        with pytest.raises(TypeError):
            ExperimentBlock()  # type: ignore[abstract]

    def test_concrete_subclass_is_instantiable(self) -> None:
        """A class implementing all three methods must be instantiable."""

        class ConcreteBlock(ExperimentBlock):
            def setup(self, config: ExperimentConfig) -> None:
                pass

            def run(self) -> None:
                pass

            def report(self) -> dict[str, Any]:
                return {}

        block = ConcreteBlock()
        assert isinstance(block, ExperimentBlock)

    def test_partial_implementation_raises(self) -> None:
        """A class missing any abstract method must raise TypeError on instantiation."""

        class PartialBlock(ExperimentBlock):
            def setup(self, config: ExperimentConfig) -> None:
                pass

            def run(self) -> None:
                pass

            # report() not implemented

        with pytest.raises(TypeError):
            PartialBlock()  # type: ignore[abstract]


class TestBlockRegistry:
    def test_discover_returns_dict(self) -> None:
        """discover() must return a dict."""
        result = BlockRegistry.discover()
        assert isinstance(result, dict)

    def test_concrete_block_discovered(self) -> None:
        """A concrete ExperimentBlock subclass must appear in discover()."""

        class MyBlock(ExperimentBlock):
            def setup(self, config: ExperimentConfig) -> None:
                pass

            def run(self) -> None:
                pass

            def report(self) -> dict[str, Any]:
                return {"status": "ok"}

        result = BlockRegistry.discover()
        assert "MyBlock" in result
        assert result["MyBlock"] is MyBlock

    def test_abstract_class_not_discovered(self) -> None:
        """Abstract classes must not appear in discover()."""
        result = BlockRegistry.discover()
        assert "ExperimentBlock" not in result
