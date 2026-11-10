"""The pair order rule lives in one place and every paired benchmark uses it."""

import pytest

from research.workflow_benchmark import (
    paired_classification,
    paired_detection,
    paired_segmentation,
    phase_contract,
)


@pytest.mark.parametrize(
    ("seed", "first"),
    [(0, "native_pytorch"), (1, "visionforge_trainer"), (36, "native_pytorch")],
)
def test_execution_order_alternates_with_seed_parity(seed: int, first: str) -> None:
    order = phase_contract.execution_order(seed)
    assert order[0] == first
    assert sorted(order) == sorted(phase_contract.EXECUTION_ORDER)


@pytest.mark.parametrize(
    "module", [paired_classification, paired_segmentation, paired_detection]
)
def test_every_paired_benchmark_uses_the_shared_rule(module: object) -> None:
    assert module.execution_order is phase_contract.execution_order  # type: ignore[attr-defined]
