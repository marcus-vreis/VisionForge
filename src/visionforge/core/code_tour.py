"""The real code of a training, cut into the sections the "Código" tab shows (ADR-173).

Nothing here is a simplified copy: each section is the source of the function
the run calls, read with ``inspect`` when asked, so it cannot drift from what
trains. A note points at lines by their text, never by number; a test fails
when a change to the code leaves a note without its line. A note with
conditions marks the branch this run's settings take.

Each task has its own tour and its own settings (``TOURS``, ``TOUR_SETTINGS``);
``tour_settings`` reads them from a request's query, keeping only the values
the task knows. Detection on the Ultralytics backend has a tour of its own
(``ULTRALYTICS_DETECTION``, ADR-194): VisionForge's layer, then the library's
installed training loop, whose notes are tolerant because its code is not ours.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import inspect
import sys
import textwrap
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from visionforge.utils.messages import tr

_FAMILIES = frozenset({"resnet", "efficientnet", "vit", "swin", "convnext", "vgg"})
_ON_OFF = frozenset({"on", "off"})

# Lines the tours anchor on, and the note keys, that more than one tour uses.
_TRAIN_MODE = "model.train()"
_EVAL_MODE = "model.eval()"
_ZERO_GRAD = "optimizer.zero_grad(set_to_none=True)"
_ZERO_GRAD_CALL = "optimizer.zero_grad"
_NO_GRAD = "with torch.no_grad():"
_FORWARD = "outputs = model(inputs)"
_ADD_LOSS = "total_loss += loss.item()"
_RAISE = "raise ValueError"
_IF_PRETRAINED = "if pretrained:"
_KEY_TRAIN_MODE = "code.train.mode"
_KEY_ZERO_GRAD = "code.train.zero_grad"
_KEY_VALIDATE_MODE = "code.validate.mode"
_KEY_VALIDATE_NO_GRAD = "code.validate.no_grad"
_KEY_DETECTION_WEIGHTS = "code.detection.weights"


@dataclass(frozen=True)
class Note:
    """An explanation of the lines from ``start`` up to (not including) ``end``.

    ``start`` is the text of the first line, and ``nth`` picks its occurrence
    when the text repeats. ``end`` is the text of the first line after the
    span, searched after ``start``; without it the note covers one line.
    ``when`` names the settings under which this is the path the run takes.
    """

    start: str
    key: str
    end: str | None = None
    nth: int = 1
    when: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Section:
    """One function of the training, shown whole, with its notes.

    ``library`` names the distribution the function belongs to when it is not
    this project's code. Such a section is read from whatever version is
    installed: a note whose line that version lacks is dropped instead of
    failing, and the section is left out when the library is absent.
    """

    id: str
    target: Callable[[], Callable[..., Any]]
    notes: tuple[Note, ...]
    library: str | None = None


def _transforms() -> Callable[..., Any]:
    from visionforge.core.data import _build_transforms

    return _build_transforms


def _final_layer() -> Callable[..., Any]:
    from visionforge.models.factory import replace_final_layer

    return replace_final_layer


def _criterion() -> Callable[..., Any]:
    from visionforge.core.trainer import Trainer

    return Trainer._build_criterion


def _train_epoch() -> Callable[..., Any]:
    from visionforge.core.trainer import Trainer

    return Trainer._train_epoch


def _eval_epoch() -> Callable[..., Any]:
    from visionforge.core.trainer import Trainer

    return Trainer._eval_epoch


def _evaluate() -> Callable[..., Any]:
    from visionforge.core.evaluator import Evaluator

    return Evaluator.evaluate


_BINARY = (("kind", "binary"),)
_MULTICLASS = (("kind", "multiclass"),)


def _family(name: str, start: str, end: str) -> Note:
    return Note(start, "code.model.family", end=end, when=(("family", name),))


CLASSIFICATION: tuple[Section, ...] = (
    Section(
        "data",
        _transforms,
        (
            Note("T.Resize(config.image_size),", "code.data.resize", end="if is_train"),
            Note(
                "if is_train and config.augment:",
                "code.data.augment",
                end="T.ToTensor(),",
                when=(("augment", "on"),),
            ),
            Note("T.ToTensor(),", "code.data.tensor", end="return T.Compose"),
        ),
    ),
    Section(
        "model",
        _final_layer,
        (
            _family("resnet", 'if name.startswith("resnet"):', "elif"),
            _family("efficientnet", 'elif name.startswith("efficientnet"):', "elif"),
            _family("vit", 'elif name.startswith("vit"):', "elif"),
            _family("swin", 'elif name.startswith("swin"):', "elif"),
            _family("convnext", 'elif name.startswith("convnext"):', "elif"),
            Note(
                'elif name.startswith("vgg") or name == "alexnet":',
                "code.model.family",
                end="clf[6] = nn.Linear",
                when=(("family", "vgg"),),
            ),
        ),
    ),
    Section(
        "loss",
        _criterion,
        (
            Note("return nn.BCEWithLogitsLoss()", "code.loss.binary", when=_BINARY),
            Note(
                "return nn.CrossEntropyLoss()", "code.loss.multiclass", when=_MULTICLASS
            ),
        ),
    ),
    Section(
        "train",
        _train_epoch,
        (
            Note(_TRAIN_MODE, _KEY_TRAIN_MODE),
            Note(
                "for inputs, labels in loader:",
                "code.train.batch",
                end='if self._config.task == "binary":',
            ),
            Note(
                "target = labels.float().unsqueeze(1)",
                "code.train.binary_target",
                when=_BINARY,
            ),
            Note(_ZERO_GRAD, _KEY_ZERO_GRAD),
            Note("if use_amp:", "code.train.amp", end="else:", when=(("amp", "on"),)),
            Note(
                _FORWARD,
                "code.train.step",
                end=_ADD_LOSS,
                nth=2,
                when=(("amp", "off"),),
            ),
            Note(_NO_GRAD, "code.train.count", end="n = len(loader)"),
            Note(
                "preds = (outputs.sigmoid() > 0.5).squeeze(1).long()",
                "code.predict.binary",
                when=_BINARY,
            ),
            Note(
                "preds = outputs.argmax(dim=1)",
                "code.predict.multiclass",
                when=_MULTICLASS,
            ),
        ),
    ),
    Section(
        "validate",
        _eval_epoch,
        (
            Note(_EVAL_MODE, _KEY_VALIDATE_MODE),
            Note(_NO_GRAD, _KEY_VALIDATE_NO_GRAD),
            Note(_ADD_LOSS, "code.validate.loss"),
        ),
    ),
    Section(
        "test",
        _evaluate,
        (
            Note(_EVAL_MODE, "code.test.once"),
            Note(
                "pos = outputs.sigmoid().squeeze(1)",
                "code.predict.binary",
                end="else:",
                when=_BINARY,
            ),
            Note(
                "softmax = outputs.softmax(dim=1)",
                "code.predict.multiclass",
                end="all_preds.extend",
                when=_MULTICLASS,
            ),
            Note(
                'avg = "binary" if self._config.task == "binary" else "macro"',
                "code.test.metrics",
                end="return EvalResult(",
            ),
        ),
    ),
)


def _regression_item() -> Callable[..., Any]:
    from visionforge.core.regression_data import RegressionCsvDataset

    return RegressionCsvDataset.__getitem__


def _regression_model() -> Callable[..., Any]:
    from visionforge.models.regression_factory import RegressionModelFactory

    return RegressionModelFactory.create


def _regression_criterion() -> Callable[..., Any]:
    from visionforge.core.regression_trainer import RegressionTrainer

    return RegressionTrainer._build_criterion


def _shared_train_epoch() -> Callable[..., Any]:
    from visionforge.core.training_loop import train_epoch

    return train_epoch


def _regression_eval_epoch() -> Callable[..., Any]:
    from visionforge.core.regression_trainer import RegressionTrainer

    return RegressionTrainer._eval_epoch


def _regression_test() -> Callable[..., Any]:
    from visionforge.core.regression_trainer import RegressionTrainer

    return RegressionTrainer.evaluate_with_predictions


REGRESSION: tuple[Section, ...] = (
    Section(
        "data",
        _regression_item,
        (
            Note(
                "with Image.open(image_path) as img:",
                "code.regression.data.image",
                end="target = torch.tensor",
            ),
            Note(
                "target = torch.tensor(targets, dtype=torch.float32)",
                "code.regression.data.target",
            ),
        ),
    ),
    Section(
        "model",
        _regression_model,
        (
            Note(
                "if config.custom_model is not None:",
                "code.regression.model.custom",
                end="use_imagenet = config.pretrained",
            ),
            Note(
                "model = build_backbone(config.name, use_imagenet=use_imagenet)",
                "code.regression.model.backbone",
            ),
            Note(
                "replace_final_layer(model, config.name, config.num_targets)",
                "code.regression.model.head",
            ),
        ),
    ),
    Section(
        "loss",
        _regression_criterion,
        (
            Note(
                "return nn.L1Loss()",
                "code.regression.loss.mae",
                when=(("loss", "mae"),),
            ),
            Note(
                "return nn.HuberLoss(delta=cfg.huber_delta)",
                "code.regression.loss.huber",
                when=(("loss", "huber"),),
            ),
            Note(
                "return nn.MSELoss()",
                "code.regression.loss.mse",
                when=(("loss", "mse"),),
            ),
        ),
    ),
    Section(
        "train",
        _shared_train_epoch,
        (
            Note(_TRAIN_MODE, _KEY_TRAIN_MODE),
            Note(
                "if validate_targets is not None:",
                "code.regression.train.hooks",
                end="inputs = inputs.to(device",
            ),
            Note(
                "inputs = inputs.to(device, non_blocking=True)",
                "code.regression.train.batch",
                end=_ZERO_GRAD_CALL,
            ),
            Note(_ZERO_GRAD, _KEY_ZERO_GRAD),
            Note(
                _FORWARD,
                "code.regression.train.step",
                end=_ADD_LOSS,
            ),
            Note(
                "return total_loss / n if n > 0 else 0.0",
                "code.regression.train.mean",
            ),
        ),
    ),
    Section(
        "validate",
        _regression_eval_epoch,
        (
            Note(_EVAL_MODE, _KEY_VALIDATE_MODE),
            Note(_NO_GRAD, _KEY_VALIDATE_NO_GRAD),
            Note(_ADD_LOSS, "code.validate.loss"),
            Note(
                "metrics.update(outputs, targets)", "code.regression.validate.metrics"
            ),
        ),
    ),
    Section(
        "test",
        _regression_test,
        (
            Note(_EVAL_MODE, "code.test.once"),
            Note(
                "metrics.update(outputs, targets)",
                "code.regression.test.collect",
                end="empty = np.zeros",
            ),
            Note("metrics.compute(),", "code.regression.test.metrics"),
        ),
    ),
)


def _segmentation_item() -> Callable[..., Any]:
    from visionforge.core.segmentation_data import SegmentationDataset

    return SegmentationDataset.__getitem__


def _segmentation_model() -> Callable[..., Any]:
    from visionforge.models.segmentation_factory import SegmentationModelFactory

    return SegmentationModelFactory._build


def _unet_forward() -> Callable[..., Any]:
    from visionforge.models.segmentation_factory import UNet

    return UNet.forward


def _segmentation_criterion() -> Callable[..., Any]:
    from visionforge.core.segmentation_trainer import SegmentationTrainer

    return SegmentationTrainer._build_criterion


def _segmentation_eval_epoch() -> Callable[..., Any]:
    from visionforge.core.segmentation_trainer import SegmentationTrainer

    return SegmentationTrainer._eval_epoch


def _segmentation_test() -> Callable[..., Any]:
    from visionforge.core.segmentation_trainer import SegmentationTrainer

    return SegmentationTrainer.evaluate_with_confusion


def _segmentation_metrics() -> Callable[..., Any]:
    from visionforge.core.segmentation_trainer import SegmentationMetricAccumulator

    return SegmentationMetricAccumulator.compute


_UNET = (("arch", "unet"),)
_TORCHVISION_SEGMENTER = (("arch", "torchvision"),)

SEGMENTATION: tuple[Section, ...] = (
    Section(
        "data",
        _segmentation_item,
        (
            Note(
                'image = self._preprocess(im.convert("RGB"))',
                "code.segmentation.data.image",
            ),
            Note(
                'if mask.mode in ("RGB", "RGBA"):',
                "code.segmentation.data.mask",
                end="size = [self._image_size, self._image_size]",
            ),
            Note(
                "image = TF.resize(image, size,",
                "code.segmentation.data.resize",
                end="if self._hflip",
            ),
            Note(
                "if self._hflip and random.random() < 0.5:",
                "code.segmentation.data.joint",
                end="if self._jitter is not None:",
            ),
            Note(
                "if self._jitter is not None:",
                "code.segmentation.data.jitter",
                end="image_t = TF.to_tensor(image)",
            ),
            Note(
                "image_t = TF.to_tensor(image)",
                "code.segmentation.data.tensor",
                end="return image_t, mask_t",
            ),
        ),
    ),
    Section(
        "model",
        _segmentation_model,
        (
            Note(
                'if name == "unet":',
                "code.segmentation.model.unet",
                end="if name in _TORCHVISION_SEG:",
                when=_UNET,
            ),
            Note(
                "if name in _TORCHVISION_SEG:",
                "code.segmentation.model.torchvision",
                end=_RAISE,
                when=_TORCHVISION_SEGMENTER,
            ),
        ),
    ),
    Section(
        "unet",
        _unet_forward,
        (
            Note(
                "c0 = self.inc(x)",
                "code.segmentation.unet.encoder",
                end="b = self.bottleneck",
                when=_UNET,
            ),
            Note(
                "b = self.bottleneck(self.pool(c3))",
                "code.segmentation.unet.bottleneck",
                when=_UNET,
            ),
            Note(
                "d3 = self.up3(self._up_cat(b, c3))",
                "code.segmentation.unet.decoder",
                end="logits: torch.Tensor",
                when=_UNET,
            ),
            Note(
                "logits: torch.Tensor = self.outc(d0)",
                "code.segmentation.unet.head",
                when=_UNET,
            ),
        ),
    ),
    Section(
        "loss",
        _segmentation_criterion,
        (
            Note(
                "return _DiceCriterion(self._ignore_index)",
                "code.segmentation.loss.dice",
                when=(("loss", "dice"),),
            ),
            Note(
                "return _CombinedLoss(self._ignore_index)",
                "code.segmentation.loss.combined",
                when=(("loss", "combined"),),
            ),
            Note(
                "return nn.CrossEntropyLoss(ignore_index=self._ignore_index)",
                "code.segmentation.loss.cross_entropy",
                when=(("loss", "cross_entropy"),),
            ),
        ),
    ),
    Section(
        "train",
        _shared_train_epoch,
        (
            Note(_TRAIN_MODE, _KEY_TRAIN_MODE),
            Note(
                "if validate_targets is not None:",
                "code.segmentation.train.check_ids",
                end="inputs = inputs.to(device",
            ),
            Note(
                "inputs = inputs.to(device, non_blocking=True)",
                "code.segmentation.train.batch",
                end=_ZERO_GRAD_CALL,
            ),
            Note(_ZERO_GRAD, _KEY_ZERO_GRAD),
            Note(
                _FORWARD,
                "code.segmentation.train.logits",
                end="loss = criterion(outputs, targets)",
            ),
            Note(
                "loss = criterion(outputs, targets)",
                "code.segmentation.train.step",
                end=_ADD_LOSS,
            ),
        ),
    ),
    Section(
        "validate",
        _segmentation_eval_epoch,
        (
            Note(_EVAL_MODE, _KEY_VALIDATE_MODE),
            Note(_NO_GRAD, _KEY_VALIDATE_NO_GRAD),
            Note(
                "check_mask_ids(targets, self._num_classes, self._ignore_index)",
                "code.segmentation.validate.check_ids",
            ),
            Note(
                "logits = segmentation_logits(model(inputs))",
                "code.segmentation.validate.logits",
            ),
            Note(
                "total_loss += criterion(logits, targets).item()",
                "code.segmentation.validate.loss",
            ),
            Note(
                "miou, dice, pixel_acc = metrics.compute()",
                "code.segmentation.validate.best",
            ),
        ),
    ),
    Section(
        "test",
        _segmentation_test,
        (
            Note(_EVAL_MODE, "code.segmentation.test.once"),
            Note(
                "per_image.append(",
                "code.segmentation.test.per_image",
                end="stacked = (",
            ),
            Note(
                "return metrics.compute(), stacked",
                "code.segmentation.test.metrics",
            ),
        ),
    ),
    Section(
        "metrics",
        _segmentation_metrics,
        (
            Note(
                "diag = cm.diag()",
                "code.segmentation.metrics.confusion",
                end="present = (row + col) > 0",
            ),
            Note(
                "present = (row + col) > 0",
                "code.segmentation.metrics.present",
            ),
            Note(
                "iou = diag[present] / union[present]",
                "code.segmentation.metrics.iou",
            ),
            Note(
                "dice = (2.0 * diag[present])",
                "code.segmentation.metrics.dice",
            ),
            Note(
                "pixel_acc = float(diag.sum().item() / total)",
                "code.segmentation.metrics.pixel_acc",
            ),
        ),
    ),
)


def _anomaly_data() -> Callable[..., Any]:
    from visionforge.core.anomaly_data import AnomalyDataModule

    return AnomalyDataModule.__init__


def _anomaly_model() -> Callable[..., Any]:
    from visionforge.models.anomaly_factory import AnomalyModelFactory

    return AnomalyModelFactory.create


def _autoencoder_epoch() -> Callable[..., Any]:
    from visionforge.core.anomaly_trainer import AnomalyTrainer

    return AnomalyTrainer._train_autoencoder_epoch


def _coreset() -> Callable[..., Any]:
    from visionforge.models.anomaly_factory import PatchCore

    return PatchCore._greedy_coreset


def _score_images() -> Callable[..., Any]:
    from visionforge.core.anomaly_trainer import score_images

    return score_images


def _patch_score() -> Callable[..., Any]:
    from visionforge.models.anomaly_factory import PatchCore

    return PatchCore.score


def _threshold() -> Callable[..., Any]:
    from visionforge.core.anomaly_trainer import compute_threshold

    return compute_threshold


def _anomaly_test() -> Callable[..., Any]:
    from visionforge.core.anomaly_trainer import AnomalyTrainer

    return AnomalyTrainer.evaluate_with_scores


def _anomaly_metrics() -> Callable[..., Any]:
    from visionforge.core.anomaly_trainer import AnomalyTrainer

    return AnomalyTrainer._metrics


_AUTOENCODER = (("method", "autoencoder"),)
_PATCHCORE = (("method", "patchcore"),)

ANOMALY: tuple[Section, ...] = (
    Section(
        "data",
        _anomaly_data,
        (
            Note(
                "train_items = [(p, 0) for p in _list_images(train_normal)]",
                "code.anomaly.data.normal_only",
            ),
            Note(
                "label = 0 if sub.name == cfg.normal_dir else 1",
                "code.anomaly.data.test_labels",
            ),
            Note(
                "self._train = AnomalyImageDataset(",
                "code.anomaly.data.train",
                end="self._test = AnomalyImageDataset(",
            ),
            Note(
                "self._calibration = AnomalyImageDataset(",
                "code.anomaly.data.calibration",
                end="self._num_workers = resolve_num_workers(",
            ),
        ),
    ),
    Section(
        "model",
        _anomaly_model,
        (
            Note(
                'if config.name == "autoencoder":',
                "code.anomaly.model.autoencoder",
                end='if config.name == "patchcore":',
                when=_AUTOENCODER,
            ),
            Note(
                'if config.name == "patchcore":',
                "code.anomaly.model.patchcore",
                end=_RAISE,
                when=_PATCHCORE,
            ),
        ),
    ),
    Section(
        "train",
        _autoencoder_epoch,
        (
            Note(_TRAIN_MODE, _KEY_TRAIN_MODE, when=_AUTOENCODER),
            Note(
                "for inputs, _ in train_loader:",
                "code.anomaly.train.batch",
                end=_ZERO_GRAD_CALL,
                when=_AUTOENCODER,
            ),
            Note(
                "recon = model(inputs)",
                "code.anomaly.train.reconstruct",
                end="loss.backward()",
                when=_AUTOENCODER,
            ),
            Note(
                "loss.backward()",
                "code.anomaly.train.step",
                end="total += loss.item()",
                when=_AUTOENCODER,
            ),
            Note(
                "return total / n if n > 0 else 0.0",
                "code.anomaly.train.best",
                when=_AUTOENCODER,
            ),
        ),
    ),
    Section(
        "coreset",
        _coreset,
        (
            Note(
                "if k >= m:",
                "code.anomaly.coreset.small",
                end="selected = [0]",
                when=_PATCHCORE,
            ),
            Note(
                "selected = [0]",
                "code.anomaly.coreset.start",
                end="step = max(1, k // 100)",
                when=_PATCHCORE,
            ),
            Note(
                "idx = int(torch.argmax(min_dist).item())",
                "code.anomaly.coreset.farthest",
                end="if progress is not None and i % step == 0:",
                when=_PATCHCORE,
            ),
            Note(
                "return features[selected].clone()",
                "code.anomaly.coreset.bank",
                when=_PATCHCORE,
            ),
        ),
    ),
    Section(
        "score",
        _score_images,
        (
            Note(
                "if isinstance(model, ConvAutoencoder):",
                "code.anomaly.score.autoencoder",
                end="if isinstance(model, PatchCore):",
                when=_AUTOENCODER,
            ),
            Note(
                "if isinstance(model, PatchCore):",
                "code.anomaly.score.patchcore",
                end="raise TypeError",
                when=_PATCHCORE,
            ),
        ),
    ),
    Section(
        "patch_score",
        _patch_score,
        (
            Note(
                "feats = self._forward_features(x)",
                "code.anomaly.patch_score.features",
                end="dists = torch.cdist",
                when=_PATCHCORE,
            ),
            Note(
                "dists = torch.cdist(flat, self._memory)",
                "code.anomaly.patch_score.nearest",
                end="return nn_dist.max",
                when=_PATCHCORE,
            ),
            Note(
                "return nn_dist.max(dim=1).values",
                "code.anomaly.patch_score.image",
                when=_PATCHCORE,
            ),
        ),
    ),
    Section(
        "threshold",
        _threshold,
        (
            Note(
                "return float(np.percentile(",
                "code.anomaly.threshold.percentile",
            ),
        ),
    ),
    Section(
        "test",
        _anomaly_test,
        (
            Note(_EVAL_MODE, "code.anomaly.test.once"),
            Note(
                "normal_scores = self._collect_scores(model, calibration_loader)[0]",
                "code.anomaly.test.calibration",
            ),
            Note(
                "test_scores, test_labels = self._collect_scores(model, test_loader)",
                "code.anomaly.test.scores",
            ),
        ),
    ),
    Section(
        "metrics",
        _anomaly_metrics,
        (
            Note(
                "auroc = compute_auroc(test_scores, test_labels)",
                "code.anomaly.metrics.auroc",
            ),
            Note(
                "threshold = compute_threshold(normal_scores, self._percentile)",
                "code.anomaly.metrics.threshold",
            ),
            Note(
                "preds = (test_scores >= threshold).long().cpu().numpy()",
                "code.anomaly.metrics.f1",
                end="return auroc, threshold, f1",
            ),
        ),
    ),
)


def _detection_item() -> Callable[..., Any]:
    from visionforge.core.detection_dataset import DetectionDataset

    return DetectionDataset.__getitem__


def _detection_labels() -> Callable[..., Any]:
    from visionforge.core.detection_dataset import DetectionDataset

    return DetectionDataset._read_labels


def _detection_collate() -> Callable[..., Any]:
    from visionforge.core.detection_dataset import detection_collate

    return detection_collate


def _detector() -> Callable[..., Any]:
    from visionforge.models.detection_factory import build_torchvision_detector

    return build_torchvision_detector


def _faster_rcnn() -> Callable[..., Any]:
    from visionforge.models.detection_factory import _build_faster_rcnn

    return _build_faster_rcnn


def _ssd() -> Callable[..., Any]:
    from visionforge.models.detection_factory import _build_ssd

    return _build_ssd


def _retinanet() -> Callable[..., Any]:
    from visionforge.models.detection_factory import _build_retinanet

    return _build_retinanet


def _detection_epoch() -> Callable[..., Any]:
    from visionforge.core.detection_trainer import DetectionTrainer

    return DetectionTrainer._run_torchvision_epoch


def _detection_val_loss() -> Callable[..., Any]:
    from visionforge.core.detection_trainer import DetectionTrainer

    return DetectionTrainer._eval_torchvision_loss


def _detection_val_map() -> Callable[..., Any]:
    from visionforge.core.detection_trainer import DetectionTrainer

    return DetectionTrainer._eval_torchvision_map


def _average_precision() -> Callable[..., Any]:
    from visionforge.core.detection_metrics import _ap_for_class

    return _ap_for_class


_FASTER_RCNN = (("arch", "faster_rcnn"),)
_SSD = (("arch", "ssd"),)
_RETINANET = (("arch", "retinanet"),)


def _head(key: str, start: str, when: tuple[tuple[str, str], ...]) -> Note:
    return Note(start, key, end="detector: nn.Module", when=when)


DETECTION: tuple[Section, ...] = (
    Section(
        "data",
        _detection_item,
        (
            Note(
                "tensor = to_tensor(image)",
                "code.detection.data.tensor",
            ),
            Note(
                "boxes, labels = self._read_labels(img_path, width, height)",
                "code.detection.data.target",
                end="return tensor, target",
            ),
        ),
    ),
    Section(
        "labels",
        _detection_labels,
        (
            Note(
                "cls, cx, cy, w, h = (float(p) for p in parts)",
                "code.detection.labels.yolo",
                end="# Clamp to the image",
            ),
            Note(
                "x1, x2 = max(0.0, x1), min(float(width), x2)",
                "code.detection.labels.clamp",
                end="boxes.append(",
            ),
            Note(
                "labels.append(int(cls) + 1)",
                "code.detection.labels.background",
            ),
        ),
    ),
    Section(
        "collate",
        _detection_collate,
        (
            Note(
                "images, targets = zip(*batch, strict=True)",
                "code.detection.collate.tuples",
                end="return images, targets",
            ),
        ),
    ),
    Section(
        "model",
        _detector,
        (
            Note(
                "if name in _FRCNN_BUILDERS:",
                "code.detection.model.faster_rcnn",
                end="if name in _SSD_BUILDERS:",
                when=_FASTER_RCNN,
            ),
            Note(
                "if name in _SSD_BUILDERS:",
                "code.detection.model.ssd",
                end='if name == "retinanet_resnet50_fpn":',
                when=_SSD,
            ),
            Note(
                'if name == "retinanet_resnet50_fpn":',
                "code.detection.model.retinanet",
                end=_RAISE,
                when=_RETINANET,
            ),
        ),
    ),
    Section(
        "faster_rcnn",
        _faster_rcnn,
        (
            Note(
                _IF_PRETRAINED,
                _KEY_DETECTION_WEIGHTS,
                end="in_features =",
                when=_FASTER_RCNN,
            ),
            _head("code.detection.faster_rcnn.head", "in_features =", _FASTER_RCNN),
        ),
    ),
    Section(
        "ssd",
        _ssd,
        (
            Note(
                _IF_PRETRAINED,
                _KEY_DETECTION_WEIGHTS,
                end="in_channels =",
                when=_SSD,
            ),
            _head("code.detection.ssd.head", "in_channels =", _SSD),
        ),
    ),
    Section(
        "retinanet",
        _retinanet,
        (
            Note(
                _IF_PRETRAINED,
                _KEY_DETECTION_WEIGHTS,
                end="num_anchors =",
                when=_RETINANET,
            ),
            _head("code.detection.retinanet.head", "num_anchors =", _RETINANET),
        ),
    ),
    Section(
        "train",
        _detection_epoch,
        (
            Note(_TRAIN_MODE, _KEY_TRAIN_MODE),
            Note(
                "images = [img.to(device) for img in images]",
                "code.detection.train.batch",
                end="loss_dict = model(images, targets)",
            ),
            Note(
                "loss_dict = model(images, targets)",
                "code.detection.train.losses",
                end=_ZERO_GRAD_CALL,
            ),
            Note(
                _ZERO_GRAD,
                "code.detection.train.step",
                end="total += float(loss.item())",
            ),
        ),
    ),
    Section(
        "validate",
        _detection_val_loss,
        (
            Note(_TRAIN_MODE, "code.detection.validate.train_mode"),
            Note(_NO_GRAD, _KEY_VALIDATE_NO_GRAD),
            Note(
                "total += float(sum(loss_dict.values()).item())",
                "code.detection.validate.loss",
            ),
        ),
    ),
    Section(
        "map",
        _detection_val_map,
        (
            # The first _EVAL_MODE is in the docstring.
            Note(_EVAL_MODE, "code.detection.map.eval", nth=2),
            Note(
                "outputs = model([img.to(device) for img in images])",
                "code.detection.map.predict",
                end="for t in targets:",
            ),
            Note(
                "return mean_average_precision_50(preds, gts).map50",
                "code.detection.map.best",
            ),
        ),
    ),
    Section(
        "metric",
        _average_precision,
        (
            Note(
                "entries.sort(key=lambda e: e[0], reverse=True)",
                "code.detection.metric.sort",
            ),
            Note(
                "ious = box_iou(box.unsqueeze(0), gt_boxes)[0]",
                "code.detection.metric.match",
                end="tp_cum = np.cumsum(tp)",
            ),
            Note(
                "recalls = tp_cum / n_gt",
                "code.detection.metric.curve",
                end="return _ap_all_points",
            ),
            Note(
                "return _ap_all_points(recalls, precisions)",
                "code.detection.metric.ap",
            ),
        ),
    ),
)

TOURS: dict[str, tuple[Section, ...]] = {
    "classification": CLASSIFICATION,
    "regression": REGRESSION,
    "segmentation": SEGMENTATION,
    "anomaly": ANOMALY,
    "detection": DETECTION,
}


def _yolo_train_kwargs() -> Callable[..., Any]:
    from visionforge.core.detection_trainer import DetectionTrainer

    return DetectionTrainer._ultralytics_train_kwargs


def _yolo_augmentation() -> Callable[..., Any]:
    from visionforge.core.detection_trainer import _augmentation_kwargs

    return _augmentation_kwargs


def _yolo_fit() -> Callable[..., Any]:
    from visionforge.core.detection_trainer import DetectionTrainer

    return DetectionTrainer._fit_ultralytics


def _yolo_epoch() -> Callable[..., Any]:
    from visionforge.core.detection_trainer import DetectionTrainer

    return DetectionTrainer._record_ultralytics_epoch


def _ultralytics_loop() -> Callable[..., Any]:
    """``BaseTrainer._do_train`` of the installed Ultralytics (AGPL-3.0)."""
    from visionforge.utils.third_party_privacy import apply_opt_outs

    apply_opt_outs()  # before ultralytics' first import in a `visionforge gui`
    with contextlib.redirect_stdout(sys.stderr):
        from ultralytics.engine.trainer import BaseTrainer

    loop: Callable[..., Any] = BaseTrainer._do_train
    return loop


_YOLO_FINISH = "return self._finish_ultralytics_training("

# What VisionForge does around Ultralytics, then the library's own loop.
ULTRALYTICS_DETECTION: tuple[Section, ...] = (
    Section(
        "yolo_args",
        _yolo_train_kwargs,
        (
            Note('"epochs": cfg.epochs,', "code.yolo.args.core", end="# optimizer"),
            Note('"box": cfg.box,', "code.yolo.args.gains", end="# regularization"),
            Note("**_augmentation_kwargs(aug),", "code.yolo.args.augmentation"),
        ),
    ),
    Section(
        "yolo_augment",
        _yolo_augmentation,
        (
            Note(
                "if not aug.augment:",
                "code.yolo.augment.neutral",
                end="return {key: getattr(aug, key)",
            ),
            Note(
                "return {key: getattr(aug, key)",
                "code.yolo.augment.config",
            ),
        ),
    ),
    Section(
        "yolo_fit",
        _yolo_fit,
        (
            Note("if YOLO is None:", "code.yolo.fit.missing", end="cfg = "),
            Note("model = YOLO(self._resolve_weights())", "code.yolo.fit.weights"),
            Note(
                "def _on_epoch_end(trainer: Any) -> None:",
                "code.yolo.fit.callback",
                end="# Ultralytics reads `trainer.stop`",
            ),
            Note("trainer.stop = True", "code.yolo.fit.stop"),
            Note("model.train(", "code.yolo.fit.train", end=_YOLO_FINISH, nth=1),
        ),
    ),
    Section(
        "yolo_epoch",
        _yolo_epoch,
        (
            Note(
                "map50 = _extract(metrics, _MAP50_KEY)",
                "code.yolo.epoch.metrics",
                end="# `trainer.metrics` only holds",
            ),
            Note(
                "train_metrics = _train_loss_metrics(trainer)",
                "code.yolo.epoch.train_loss",
            ),
            Note(
                "history[:] = [h for h in history if h.epoch < epoch]",
                "code.yolo.epoch.resume",
            ),
        ),
    ),
    Section(
        "yolo_loop",
        _ultralytics_loop,
        (
            Note("self._setup_train()", "code.yolo.loop.setup"),
            Note("while True:", "code.yolo.loop.epochs"),
            Note("self.scheduler.step()", "code.yolo.loop.scheduler"),
            Note("# Warmup", "code.yolo.loop.warmup", end="# Forward"),
            Note(
                "with autocast(self.amp):", "code.yolo.loop.forward", end="# Backward"
            ),
            Note("self.scaler.scale(self.loss).backward()", "code.yolo.loop.backward"),
            Note("self.optimizer_step()", "code.yolo.loop.step"),
            Note(
                "self.metrics, self.fitness = self.validate()",
                "code.yolo.loop.validate",
            ),
            Note('self.run_callbacks("on_fit_epoch_end")', "code.yolo.loop.callback"),
            Note("self.final_eval()", "code.yolo.loop.final_eval"),
        ),
        library="ultralytics",
    ),
)

# Settings a note can depend on, per task, and their values.
TOUR_SETTINGS: dict[str, dict[str, frozenset[str]]] = {
    "classification": {
        "kind": frozenset({"binary", "multiclass"}),
        "amp": _ON_OFF,
        "augment": _ON_OFF,
        "family": _FAMILIES,
    },
    "regression": {"loss": frozenset({"mse", "mae", "huber"})},
    "segmentation": {
        "loss": frozenset({"cross_entropy", "dice", "combined"}),
        "arch": frozenset({"unet", "torchvision"}),
    },
    "anomaly": {"method": frozenset({"autoencoder", "patchcore"})},
    "detection": {
        "backend": frozenset({"torchvision", "ultralytics"}),
        "arch": frozenset({"faster_rcnn", "ssd", "retinanet"}),
    },
}

# Query flags as the page sends them, turned into the settings' on/off.
_FLAGS = {"true": "on", "false": "off"}


def model_family(model_name: str) -> str | None:
    """The branch of ``replace_final_layer`` a model name takes, if any."""
    if model_name == "alexnet":
        return "vgg"
    for family in sorted(_FAMILIES):
        if model_name.startswith(family):
            return family
    return None


def segmentation_arch(model_name: str) -> str | None:
    """The branch of ``SegmentationModelFactory._build`` a model name takes, if any."""
    from visionforge.models.segmentation_factory import _TORCHVISION_SEG

    if model_name == "unet":
        return "unet"
    return "torchvision" if model_name in _TORCHVISION_SEG else None


def detection_arch(model_name: str) -> str | None:
    """The builder ``build_torchvision_detector`` picks for a model name, if any."""
    from visionforge.models.detection_factory import _FRCNN_BUILDERS, _SSD_BUILDERS

    if model_name in _FRCNN_BUILDERS:
        return "faster_rcnn"
    if model_name in _SSD_BUILDERS:
        return "ssd"
    return "retinanet" if model_name == "retinanet_resnet50_fpn" else None


# The setting a task reads from the model name, and how.
_FROM_MODEL: dict[str, tuple[str, Callable[[str], str | None]]] = {
    "classification": ("family", model_family),
    "segmentation": ("arch", segmentation_arch),
    "detection": ("arch", detection_arch),
}


def tour_sections(task: str, settings: Mapping[str, str]) -> tuple[Section, ...]:
    """The sections ``task`` shows for these settings.

    Detection on Ultralytics, the default backend, shows its own tour: a run
    that does not say torchvision gets it too (ADR-194).

    Raises:
        KeyError: a task without a tour.
    """
    if task == "detection" and settings.get("backend") != "torchvision":
        return ULTRALYTICS_DETECTION
    return TOURS[task]


def tour_settings(task: str, query: Mapping[str, str]) -> dict[str, str]:
    """The settings of ``task`` a request's query decides.

    A key the task does not know, or a value it does not take, is dropped, so
    its notes stay undecided. ``true``/``false`` read as ``on``/``off``;
    ``model`` is a model name, turned into the setting it decides (the
    classification family, the segmentation architecture, the detector).

    Raises:
        KeyError: a task without a tour.
    """
    known = TOUR_SETTINGS[task]
    settings: dict[str, str] = {}
    for key, raw in query.items():
        value: str | None
        if key == "model" and task in _FROM_MODEL:
            key, read = _FROM_MODEL[task]
            value = read(raw)
        else:
            value = _FLAGS.get(raw, raw)
        if value is not None and value in known.get(key, frozenset()):
            settings[key] = value
    return settings


def code_tour(task: str, settings: Mapping[str, str]) -> list[dict[str, Any]]:
    """The sections of ``task`` with their source and notes, in the request's language.

    ``settings`` are what the run uses (``TOUR_SETTINGS[task]``); one missing
    or unknown leaves its notes undecided.

    Raises:
        KeyError: a task without a tour.
    """
    allowed = TOUR_SETTINGS[task]
    known = {
        key: value
        for key, value in settings.items()
        if key in allowed and value in allowed[key]
    }
    sections = (_section(s, known) for s in tour_sections(task, known))
    return [section for section in sections if section is not None]


def _section(section: Section, settings: Mapping[str, str]) -> dict[str, Any] | None:
    """The section as the page reads it; None for a library that is not here."""
    if section.library is None:
        target = section.target()
    else:
        try:
            target = section.target()
            library = _library(section.library)
        except (ImportError, AttributeError):  # PackageNotFoundError is an ImportError
            return None
    raw, first_line = inspect.getsourcelines(target)
    lines = textwrap.dedent("".join(raw)).rstrip("\n").split("\n")
    if section.library is None:
        notes = [_note(note, lines, settings) for note in section.notes]
    else:
        notes = _tolerant_notes(section.notes, lines, settings)
    item: dict[str, Any] = {
        "id": section.id,
        "name": target.__qualname__,
        "file": _module_file(target),
        "first_line": first_line,
        "lines": lines,
        "notes": notes,
        "notes_resolved": len(notes),
        "notes_total": len(section.notes),
    }
    if section.library is not None:
        item["library"] = library
    return item


def _tolerant_notes(
    notes: tuple[Note, ...], lines: list[str], settings: Mapping[str, str]
) -> list[dict[str, Any]]:
    """The notes that still find their lines in this version of the library."""
    resolved = []
    for note in notes:
        with contextlib.suppress(ValueError):
            resolved.append(_note(note, lines, settings))
    return resolved


def _library(distribution: str) -> dict[str, str]:
    """Name, installed version and declared license of a third-party library."""
    meta = importlib.metadata.metadata(distribution)
    module = sys.modules.get(distribution.replace("-", "_"))
    version = getattr(module, "__version__", None) or meta["Version"]
    declared = meta.get("License-Expression") or meta.get("License") or ""
    return {"name": meta["Name"], "version": str(version), "license": declared}


def _note(note: Note, lines: list[str], settings: Mapping[str, str]) -> dict[str, Any]:
    start, end = note_span(note, lines)
    family = dict(note.when).get("family")
    return {
        "first": start,
        "last": end,
        "text": tr(note.key, family=family) if family else tr(note.key),
        "active": _active(note, settings),
    }


def note_span(note: Note, lines: list[str]) -> tuple[int, int]:
    """``(first, last)`` 0-based line indexes the note covers, inclusive.

    Raises:
        ValueError: the start text is not found ``nth`` times, or the end
            text is not found after it.
    """
    hits = [i for i, line in enumerate(lines) if note.start in line]
    if len(hits) < note.nth:
        raise ValueError(f"Note start not found: {note.start!r}")
    start = hits[note.nth - 1]
    if note.end is None:
        return start, start
    for index in range(start + 1, len(lines)):
        if note.end in lines[index]:
            return start, index - 1
    raise ValueError(f"Note end not found after {note.start!r}: {note.end!r}")


def _active(note: Note, settings: Mapping[str, str]) -> bool | None:
    """True on this run's path, False off it, None when unconditional or unknown."""
    if not note.when:
        return None
    if any(key not in settings for key, _ in note.when):
        return None
    return all(settings[key] == value for key, value in note.when)


def _module_file(target: Callable[..., Any]) -> str:
    """The file inside the package, never the absolute path of this install."""
    module = inspect.getmodule(target)
    name = module.__name__ if module else ""
    return Path(*name.split(".")).with_suffix(".py").as_posix()
