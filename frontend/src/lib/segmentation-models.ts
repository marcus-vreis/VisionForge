/** Segmentation model options + form state — mirrors the Python
 *  SegmentationModelConfig backbones in visionforge/utils/segmentation_config.py.
 *  Keep in sync: a name the backend rejects produces a 422 on submit. */

import type { PreprocessingStep } from "../components/PreprocessingPanel";
import {
  mergeFormShape,
  stepsFromPayload,
  transferFromPayload,
  transformsFormFromPayload,
} from "./form-import";
import {
  buildPreprocessingPayload,
  buildTransformsPayload,
  makeDefaultTransformsForm,
  type TransformsForm,
} from "./transforms-form";

export interface SegmentationModelOption {
  value: string;
  label: string;
  sub?: string;
}

export const SEGMENTATION_MODELS: SegmentationModelOption[] = [
  { value: "deeplabv3_resnet50", label: "DeepLabV3 · R50", sub: "imagenet · 42M" },
  { value: "deeplabv3_resnet101", label: "DeepLabV3 · R101", sub: "deep · 61M" },
  {
    value: "deeplabv3_mobilenet_v3_large",
    label: "DeepLabV3 · MobileNetV3",
    sub: "leve · 11M",
  },
  { value: "fcn_resnet50", label: "FCN · R50", sub: "32.9M" },
  { value: "fcn_resnet101", label: "FCN · R101", sub: "51.9M" },
  {
    value: "lraspp_mobilenet_v3_large",
    label: "LR-ASPP · MobileNetV3",
    sub: "mobile · 3.2M",
  },
  { value: "unet", label: "U-Net", sub: "clássico · 31M" },
];

export const SEGMENTATION_LOSSES: { value: string; label: string }[] = [
  { value: "cross_entropy", label: "Cross-Entropy" },
  { value: "dice", label: "Dice" },
  { value: "combined", label: "CE + Dice" },
];

export type TransferMode = "none" | "feature_extraction" | "fine_tuning";

/** Controlled form state for a segmentation run — mirrors SegmentationConfig. */
export interface SegmentationForm {
  name: string;
  model: {
    name: string;
    num_classes: number;
    pretrained: boolean;
  };
  /** Transfer learning over the backbone; "none" → full training. Meaningful for
   *  the torchvision families (ImageNet-pretrained); U-Net has no pretrained weights. */
  transfer: TransferMode;
  backbone_lr_multiplier: number;
  data: {
    base_dir: string;
    images_subdir: string;
    masks_subdir: string;
    train_dir: string;
    val_dir: string;
    test_dir: string;
    image_size: number;
    ignore_index: number;
  };
  training: {
    epochs: number;
    batch_size: number;
    learning_rate: number;
    loss: string;
    optimizer: string;
    early_stopping_patience: number;
    seed: number;
  };
  /** Filter pipeline applied before augmentation (data.preprocessing.steps). */
  preprocessing: PreprocessingStep[];
  /** Augmentation/normalization (data.transforms) — joint image+mask hflip/
   *  rotation, image-only jitter. ADR-059 brick A. */
  transforms: TransformsForm;
}

export function makeDefaultSegmentationForm(): SegmentationForm {
  return {
    name: "segmentation_001",
    model: { name: "deeplabv3_resnet50", num_classes: 2, pretrained: true },
    transfer: "none",
    backbone_lr_multiplier: 0.1,
    data: {
      base_dir: "",
      images_subdir: "images",
      masks_subdir: "masks",
      train_dir: "train",
      val_dir: "val",
      test_dir: "test",
      image_size: 512,
      ignore_index: 255,
    },
    training: {
      epochs: 50,
      batch_size: 8,
      learning_rate: 0.001,
      loss: "cross_entropy",
      optimizer: "adam",
      early_stopping_patience: 10,
      seed: 42,
    },
    preprocessing: [],
    transforms: makeDefaultTransformsForm(),
  };
}

/** True when ignore_index collides with a real class id (0..num_classes-1).
 *  Mirrors the backend's top-level coherence validator so the UI can warn
 *  before submit instead of bouncing off a 422. */
export function ignoreIndexCollides(form: SegmentationForm): boolean {
  const idx = form.data.ignore_index;
  return idx >= 0 && idx < form.model.num_classes;
}

/** Project the form into the SegmentationConfig wire payload. */
export function buildSegmentationPayload(
  form: SegmentationForm,
): Record<string, unknown> {
  return {
    name: form.name,
    model: { ...form.model, num_classes: Math.max(form.model.num_classes, 1) },
    data: {
      ...form.data,
      // Resize is owned by data.image_size for segmentation (joint transforms);
      // transforms carries the augmentation flags + normalization only.
      transforms: buildTransformsPayload(form.transforms),
      preprocessing: buildPreprocessingPayload(form.preprocessing),
    },
    training: { ...form.training },
    transfer_learning: buildTransferLearning(form),
  };
}

/** Map the transfer-learning form fields to the SegmentationConfig payload shape.
 *  "none" → null (full training); fine_tuning carries the backbone LR multiplier. */
export function buildTransferLearning(
  form: SegmentationForm,
): Record<string, unknown> | null {
  if (form.transfer === "none") return null;
  if (form.transfer === "feature_extraction") {
    return { mode: "feature_extraction" };
  }
  return {
    mode: "fine_tuning",
    backbone_lr_multiplier: form.backbone_lr_multiplier,
  };
}

/** Rebuild the form from an imported SegmentationConfig payload (YAML import).
 *  Inverse of buildSegmentationPayload; unknown/mistyped values fall back to
 *  the defaults instead of corrupting the form. */
export function segmentationFormFromPayload(
  payload: Record<string, unknown>,
): SegmentationForm {
  const form = mergeFormShape(makeDefaultSegmentationForm(), payload);
  const data = (payload.data ?? {}) as Record<string, unknown>;

  form.transforms = transformsFormFromPayload(data.transforms);
  form.preprocessing = stepsFromPayload(data.preprocessing);

  const tl = transferFromPayload(payload.transfer_learning);
  form.transfer = tl.transfer;
  if (tl.backbone_lr_multiplier !== null) {
    form.backbone_lr_multiplier = tl.backbone_lr_multiplier;
  }
  return form;
}
