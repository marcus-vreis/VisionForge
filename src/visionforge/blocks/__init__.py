"""Experiment blocks.

The public names are resolved on first access (PEP 562), so importing one
block module does not import every other block and its torch dependencies
(GUI-11, #202).
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from visionforge.blocks.base import ExperimentBlock
    from visionforge.blocks.batch_prediction import BatchPredictionBlock
    from visionforge.blocks.classification import ClassificationBlock
    from visionforge.blocks.cross_validation import CrossValidationBlock
    from visionforge.blocks.export_onnx import ExportONNXBlock
    from visionforge.blocks.grid_search import GridSearchBlock
    from visionforge.blocks.model_comparison import ModelComparisonBlock
    from visionforge.blocks.random_search import RandomSearchBlock
    from visionforge.blocks.registry import BlockRegistry
    from visionforge.blocks.transfer_learning import TransferLearningBlock

_LAZY: dict[str, str] = {
    "ExperimentBlock": "visionforge.blocks.base",
    "BatchPredictionBlock": "visionforge.blocks.batch_prediction",
    "ClassificationBlock": "visionforge.blocks.classification",
    "CrossValidationBlock": "visionforge.blocks.cross_validation",
    "ExportONNXBlock": "visionforge.blocks.export_onnx",
    "GridSearchBlock": "visionforge.blocks.grid_search",
    "ModelComparisonBlock": "visionforge.blocks.model_comparison",
    "RandomSearchBlock": "visionforge.blocks.random_search",
    "BlockRegistry": "visionforge.blocks.registry",
    "TransferLearningBlock": "visionforge.blocks.transfer_learning",
}

__all__ = [
    "ExperimentBlock",
    "BatchPredictionBlock",
    "ClassificationBlock",
    "CrossValidationBlock",
    "ExportONNXBlock",
    "GridSearchBlock",
    "ModelComparisonBlock",
    "RandomSearchBlock",
    "BlockRegistry",
    "TransferLearningBlock",
]


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
