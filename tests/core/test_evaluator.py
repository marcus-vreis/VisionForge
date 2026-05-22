from pathlib import Path

import pytest
import torch
import torch.nn as nn

from visionforge.core.evaluator import EvalResult, Evaluator
from visionforge.utils.config import ExperimentConfig


class AlwaysZeroModel(nn.Module):
    """Model that always returns zero logit — predicts the negative class."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.size(0), 1)


class AlwaysOneModel(nn.Module):
    """Model that always returns a high logit — predicts the positive class."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ones(x.size(0), 1) * 10.0


def make_binary_loader(
    inputs: torch.Tensor, labels: torch.Tensor
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [(inputs, labels)]


@pytest.fixture
def binary_config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "name": "eval_test",
            "task": "binary",
            "model": {"name": "resnet18", "num_classes": 1, "pretrained": False},
            "training": {
                "learning_rate": 0.001,
                "epochs": 1,
                "batch_size": 4,
                "seed": 0,
            },
            "data": {"base_dir": str(tmp_path)},
        }
    )


@pytest.fixture
def multiclass_config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "name": "eval_test_mc",
            "task": "multiclass",
            "model": {"name": "resnet18", "num_classes": 3, "pretrained": False},
            "training": {
                "learning_rate": 0.001,
                "epochs": 1,
                "batch_size": 4,
                "seed": 0,
            },
            "data": {"base_dir": str(tmp_path)},
        }
    )


class TestEvaluatorBinary:
    def test_perfect_predictions_accuracy_is_one(
        self, binary_config: ExperimentConfig
    ) -> None:
        """Model predicting all positives on all-positive labels must yield accuracy=1."""
        inputs = torch.zeros(8, 3, 32, 32)
        labels = torch.ones(8, dtype=torch.long)
        loader = make_binary_loader(inputs, labels)

        result = Evaluator(binary_config).evaluate(AlwaysOneModel(), loader)
        assert result.accuracy == pytest.approx(1.0)

    def test_all_wrong_accuracy_is_zero(self, binary_config: ExperimentConfig) -> None:
        """Model predicting negative on all-positive labels must yield accuracy=0."""
        inputs = torch.zeros(8, 3, 32, 32)
        labels = torch.ones(8, dtype=torch.long)
        loader = make_binary_loader(inputs, labels)

        result = Evaluator(binary_config).evaluate(AlwaysZeroModel(), loader)
        assert result.accuracy == pytest.approx(0.0)

    def test_auc_roc_is_set_for_binary(self, binary_config: ExperimentConfig) -> None:
        """auc_roc must be a float for binary tasks when both classes are present."""
        inputs = torch.zeros(8, 3, 32, 32)
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
        loader = make_binary_loader(inputs, labels)

        result = Evaluator(binary_config).evaluate(AlwaysOneModel(), loader)
        assert result.auc_roc is not None

    def test_return_type_is_eval_result(self, binary_config: ExperimentConfig) -> None:
        """evaluate() must return an EvalResult instance."""
        inputs = torch.zeros(4, 3, 32, 32)
        labels = torch.zeros(4, dtype=torch.long)
        loader = make_binary_loader(inputs, labels)

        result = Evaluator(binary_config).evaluate(AlwaysZeroModel(), loader)
        assert isinstance(result, EvalResult)

    def test_confusion_matrix_shape(self, binary_config: ExperimentConfig) -> None:
        """Binary confusion matrix must be 2×2."""
        inputs = torch.zeros(8, 3, 32, 32)
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
        loader = make_binary_loader(inputs, labels)

        result = Evaluator(binary_config).evaluate(AlwaysZeroModel(), loader)
        assert len(result.confusion_matrix) == 2
        assert len(result.confusion_matrix[0]) == 2


class TestEvaluatorMulticlass:
    def test_auc_roc_is_none_for_multiclass(
        self, multiclass_config: ExperimentConfig
    ) -> None:
        """auc_roc must be None for multiclass tasks."""

        class MultiModel(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out = torch.zeros(x.size(0), 3)
                out[:, 0] = 10.0  # always predicts class 0
                return out

        inputs = torch.zeros(6, 3, 32, 32)
        labels = torch.zeros(6, dtype=torch.long)
        loader = make_binary_loader(inputs, labels)

        result = Evaluator(multiclass_config).evaluate(MultiModel(), loader)
        assert result.auc_roc is None

    def test_auc_roc_computed_when_multiple_classes_present(
        self, multiclass_config: ExperimentConfig
    ) -> None:
        """When y_true has >1 class, multiclass AUC must be computed (not None)."""

        class MultiModel(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Diverse predictions so AUC computation is well-defined.
                out = torch.zeros(x.size(0), 3)
                # Bias the prediction towards the true label so AUC > 0.5.
                for i in range(x.size(0)):
                    out[i, i % 3] = 5.0
                return out

        inputs = torch.zeros(9, 3, 32, 32)
        labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=torch.long)
        loader = make_binary_loader(inputs, labels)

        result = Evaluator(multiclass_config).evaluate(MultiModel(), loader)
        assert result.auc_roc is not None
        assert 0.0 <= result.auc_roc <= 1.0

    def test_y_proba_full_preserved(self, multiclass_config: ExperimentConfig) -> None:
        """y_proba_full must hold per-sample full distributions for downstream plots."""

        class MultiModel(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.randn(x.size(0), 3)

        inputs = torch.zeros(6, 3, 32, 32)
        labels = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)
        loader = make_binary_loader(inputs, labels)

        result = Evaluator(multiclass_config).evaluate(MultiModel(), loader)
        assert len(result.y_proba_full) == 6
        assert all(len(row) == 3 for row in result.y_proba_full)

    def test_report_is_string(self, multiclass_config: ExperimentConfig) -> None:
        """report field must be a non-empty string."""

        class MultiModel(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out = torch.zeros(x.size(0), 3)
                out[:, 0] = 10.0
                return out

        inputs = torch.zeros(6, 3, 32, 32)
        labels = torch.zeros(6, dtype=torch.long)
        loader = make_binary_loader(inputs, labels)

        result = Evaluator(multiclass_config).evaluate(MultiModel(), loader)
        assert isinstance(result.report, str)
        assert len(result.report) > 0
