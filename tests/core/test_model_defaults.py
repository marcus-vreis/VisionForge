"""The suggestions that keep a first run from failing, per architecture family.

Every number here was measured on the same data and seed (ADR-099/100), not
taken from a paper: the attention families collapse or stall at 1e-3 exactly the
way VGG and AlexNet do, and the fix is the same shape.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest
from PIL import Image

from visionforge.core import learning_rate
from visionforge.core.image_size import median_image_side, suggested_image_size
from visionforge.core.learning_rate import (
    CollapseEvidence,
    collapse_evidence,
    is_collapse_prone,
    suggested_learning_rate,
    suggested_optimizer,
)


class TestLearningRate:
    @pytest.mark.parametrize("arch", ["vit_b_16", "swin_t", "convnext_tiny"])
    def test_attention_families_get_the_measured_rate(self, arch: str) -> None:
        """swin_t collapsed at 1e-3 (0.25, one class) and reached 0.88 at 1e-4."""
        assert suggested_learning_rate(arch, "adamw") == 1e-4

    @pytest.mark.parametrize("arch", ["vgg16", "vgg19", "alexnet"])
    def test_the_pre_batchnorm_cnns_too(self, arch: str) -> None:
        assert suggested_learning_rate(arch, "adam") == 1e-4

    @pytest.mark.parametrize("arch", ["resnet18", "resnet50", "efficientnet_b1"])
    def test_normalized_cnns_keep_the_familiar_rate(self, arch: str) -> None:
        assert suggested_learning_rate(arch, "adam") == 1e-3

    @pytest.mark.parametrize("arch", ["resnet50", "vgg16", "vit_b_16"])
    def test_plain_sgd_needs_a_bigger_step_whatever_the_model(self, arch: str) -> None:
        """The trainer builds SGD without momentum, where 1e-3 barely moves."""
        assert suggested_learning_rate(arch, "sgd") == 1e-2

    @pytest.mark.parametrize("arch", ["vit_b_16", "swin_t", "convnext_tiny"])
    def test_attention_families_are_fine_tuned_with_adamw(self, arch: str) -> None:
        assert suggested_optimizer(arch) == "adamw"

    def test_convolutional_families_stay_on_adam(self) -> None:
        assert suggested_optimizer("resnet18") == "adam"


class TestCollapseProne:
    @pytest.mark.parametrize("arch", ["vgg16", "alexnet", "swin_t", "convnext_tiny"])
    def test_the_measured_failures_are_flagged(self, arch: str) -> None:
        assert is_collapse_prone(arch, "adam", 1e-3) is True

    def test_a_safe_pair_is_not(self) -> None:
        assert is_collapse_prone("resnet50", "adam", 1e-3) is False

    def test_lowering_the_rate_clears_it(self) -> None:
        assert is_collapse_prone("vgg16", "adam", 1e-4) is False

    def test_sgd_is_not_the_failure_mode(self) -> None:
        """VGG+SGD reached 0.80 at the same rate that collapsed under Adam."""
        assert is_collapse_prone("vgg16", "sgd", 1e-2) is False


class TestCollapseEvidence:
    """What the interface may quote about a model is what was measured, no more."""

    @staticmethod
    def _adam_cell(model: str) -> str:
        """The 'Adam 1e-3' cell of ``model``'s row in the module docstring tables."""
        for line in (learning_rate.__doc__ or "").splitlines():
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if line.startswith("|") and cells[0] == model:
                return cells[1]
        raise AssertionError(f"{model} has no row in the learning_rate docstring")

    @pytest.mark.parametrize(
        ("arch", "measured_on", "accuracy", "outcome"),
        [
            ("vgg16", "vgg16", 0.25, "collapse"),
            ("vgg19", "vgg16", 0.25, "collapse"),
            ("alexnet", "alexnet", 0.25, "collapse"),
            ("swin_t", "swin_t", 0.25, "collapse"),
            ("swin_v2_b", "swin_t", 0.25, "collapse"),
            ("convnext_base", "convnext_tiny", 0.25, "collapse"),
            ("vit_b_16", "vit_b_16", 0.41, "fails_to_learn"),
            ("vit_l_16", "vit_b_16", 0.41, "fails_to_learn"),
        ],
    )
    def test_each_family_cites_the_one_model_that_was_run(
        self, arch: str, measured_on: str, accuracy: float, outcome: str
    ) -> None:
        evidence = collapse_evidence(arch)

        assert evidence is not None
        assert isinstance(evidence, CollapseEvidence)
        assert (evidence.measured_on, evidence.accuracy, evidence.outcome) == (
            measured_on,
            accuracy,
            outcome,
        )

    @pytest.mark.parametrize("arch", ["maxvit_t", "resnet50", "efficientnet_b1", ""])
    def test_no_measurement_no_evidence(self, arch: str) -> None:
        assert collapse_evidence(arch) is None

    def test_the_architecture_name_is_matched_ignoring_case(self) -> None:
        assert collapse_evidence("VGG16") == collapse_evidence("vgg16")

    def test_every_evidence_row_matches_the_docstring_table(self) -> None:
        """The prose table and the data cannot drift apart unnoticed."""
        for evidence in learning_rate._EVIDENCE.values():
            cell = self._adam_cell(evidence.measured_on)

            number = re.search(r"\d\.\d+", cell)
            assert number is not None, cell
            assert float(number.group()) == evidence.accuracy
            assert ("collapse" in cell) == (evidence.outcome == "collapse")

    def test_evidence_only_exists_where_the_note_is_raised(self) -> None:
        """A measured model must be collapse-prone, or the evidence is orphaned."""
        for evidence in learning_rate._EVIDENCE.values():
            assert is_collapse_prone(evidence.measured_on, "adam", 1e-3) is True

    def test_the_evidence_is_immutable(self) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            collapse_evidence("vgg16").accuracy = 0.99  # type: ignore[misc,union-attr]


class TestImageSize:
    @staticmethod
    def _dataset(tmp_path: Path, side: int, n: int = 6) -> Path:
        d = tmp_path / "imgs"
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            Image.new("RGB", (side, side), (120, 120, 120)).save(d / f"{i}.png")
        return tmp_path

    def test_it_reads_the_dataset(self, tmp_path: Path) -> None:
        root = self._dataset(tmp_path, 32)

        assert median_image_side(root) == 32

    def test_tiny_images_are_not_upscaled_to_224(self, tmp_path: Path) -> None:
        """CIFAR-sized data at 224 is a sevenfold upscale for invented pixels."""
        root = self._dataset(tmp_path, 32)

        assert suggested_image_size(root, "resnet18") == 64

    def test_large_images_stop_at_the_pretrained_scale(self, tmp_path: Path) -> None:
        root = self._dataset(tmp_path, 800)

        assert suggested_image_size(root, "resnet18") == 224

    def test_attention_models_pin_the_size_whatever_the_data(
        self, tmp_path: Path
    ) -> None:
        """Their position embeddings are fixed; another size is an error."""
        root = self._dataset(tmp_path, 32)

        assert suggested_image_size(root, "vit_b_16") == 224
        assert suggested_image_size(root, "swin_t") == 224

    def test_no_images_means_no_opinion(self, tmp_path: Path) -> None:
        assert suggested_image_size(tmp_path, "resnet18") is None
