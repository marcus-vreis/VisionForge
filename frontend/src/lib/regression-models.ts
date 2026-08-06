/** Regression model options + form state — mirrors the Python
 *  RegressionModelConfig backbones in visionforge/utils/regression_config.py.
 *  Keep in sync: a name the backend rejects produces a 422 on submit. */

import type { PreprocessingStep } from "../components/PreprocessingPanel";
import {
  joinList,
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

export interface RegressionModelOption {
  value: string;
  label: string;
  sub?: string;
}

export const REGRESSION_MODELS: RegressionModelOption[] = [
  { value: "resnet18", label: "ResNet-18", sub: "light · 11.7M" },
  { value: "resnet34", label: "ResNet-34", sub: "21.8M" },
  { value: "resnet50", label: "ResNet-50", sub: "imagenet · 25.6M" },
  { value: "resnet101", label: "ResNet-101", sub: "deep · 44.5M" },
  { value: "efficientnet_b1", label: "EfficientNet-B1", sub: "eficiente · 7.8M" },
  { value: "efficientnet_b7", label: "EfficientNet-B7", sub: "grande · 66M" },
  { value: "vgg16", label: "VGG-16", sub: "clássico · 138M" },
  { value: "vgg19", label: "VGG-19", sub: "144M" },
  { value: "alexnet", label: "AlexNet", sub: "baseline · 61M" },
];

export const REGRESSION_LOSSES: { value: string; label: string }[] = [
  { value: "mse", label: "MSE" },
  { value: "mae", label: "MAE" },
  { value: "huber", label: "Huber" },
];

export type TransferMode = "none" | "feature_extraction" | "fine_tuning";

/** Controlled form state for a regression run — mirrors RegressionConfig. */
export interface RegressionForm {
  name: string;
  model: {
    name: string;
    num_targets: number;
    pretrained: boolean;
  };
  /** Transfer learning over the shared backbone; "none" → full training. */
  transfer: TransferMode;
  backbone_lr_multiplier: number;
  data: {
    base_dir: string;
    images_dir: string;
    train_csv: string;
    val_csv: string;
    test_csv: string;
    image_column: string;
    /** Comma-separated in the UI; split into a list for the payload. */
    target_columns: string;
    image_size: number;
  };
  training: {
    epochs: number;
    batch_size: number;
    learning_rate: number;
    loss: string;
    optimizer: string;
    early_stopping_patience: number;
    seed: number;
    deterministic: boolean;
  };
  /** Filter pipeline applied before augmentation (data.preprocessing.steps). */
  preprocessing: PreprocessingStep[];
  /** Augmentation/normalization (data.transforms) — ADR-059. */
  transforms: TransformsForm;
}

export function makeDefaultRegressionForm(): RegressionForm {
  return {
    name: "regression_001",
    model: { name: "resnet50", num_targets: 1, pretrained: true },
    transfer: "none",
    backbone_lr_multiplier: 0.1,
    data: {
      base_dir: "",
      images_dir: "images",
      train_csv: "train.csv",
      val_csv: "val.csv",
      test_csv: "test.csv",
      image_column: "image",
      target_columns: "target",
      image_size: 224,
    },
    training: {
      epochs: 50,
      batch_size: 32,
      learning_rate: 0.001,
      loss: "mse",
      optimizer: "adam",
      early_stopping_patience: 10,
      seed: 42,
      deterministic: false,
    },
    preprocessing: [],
    transforms: makeDefaultTransformsForm(),
  };
}

/** Split the comma-separated target columns into a trimmed, non-empty list. */
export function parseTargetColumns(raw: string): string[] {
  return raw
    .split(",")
    .map((c) => c.trim())
    .filter(Boolean);
}

/** Project the form into the RegressionConfig wire payload. The data section's
 *  `target_columns` is parsed from the comma-separated UI string into a list,
 *  and `num_targets` is forced to match so the backend's coherence validator
 *  passes. */
export function buildRegressionPayload(
  form: RegressionForm,
): Record<string, unknown> {
  const targets = parseTargetColumns(form.data.target_columns);
  return {
    name: form.name,
    model: { ...form.model, num_targets: Math.max(targets.length, 1) },
    data: {
      base_dir: form.data.base_dir,
      images_dir: form.data.images_dir,
      train_csv: form.data.train_csv,
      val_csv: form.data.val_csv,
      test_csv: form.data.test_csv,
      image_column: form.data.image_column,
      target_columns: targets.length > 0 ? targets : ["target"],
      // RegressionDataConfig keeps the resize under transforms.image_size (read by
      // _build_transforms); a top-level data.image_size is silently dropped.
      transforms: buildTransformsPayload(form.transforms, form.data.image_size),
      preprocessing: buildPreprocessingPayload(form.preprocessing),
    },
    training: { ...form.training },
    transfer_learning: buildTransferLearning(form),
  };
}

/** Map the transfer-learning form fields to the RegressionConfig payload shape.
 *  "none" → null (full training); fine_tuning carries the backbone LR multiplier. */
export function buildTransferLearning(
  form: RegressionForm,
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

/** Rebuild the form from an imported RegressionConfig payload (YAML import).
 *  Inverse of buildRegressionPayload; unknown/mistyped values fall back to
 *  the defaults instead of corrupting the form. */
export function regressionFormFromPayload(
  payload: Record<string, unknown>,
): RegressionForm {
  const form = mergeFormShape(makeDefaultRegressionForm(), payload);
  const data = (payload.data ?? {}) as Record<string, unknown>;
  const transforms = data.transforms as Record<string, unknown> | undefined;

  form.data.target_columns =
    joinList(data.target_columns) ?? form.data.target_columns;
  if (transforms && typeof transforms.image_size === "number") {
    form.data.image_size = transforms.image_size;
  }
  form.transforms = transformsFormFromPayload(transforms);
  form.preprocessing = stepsFromPayload(data.preprocessing);

  const tl = transferFromPayload(payload.transfer_learning);
  form.transfer = tl.transfer;
  if (tl.backbone_lr_multiplier !== null) {
    form.backbone_lr_multiplier = tl.backbone_lr_multiplier;
  }
  return form;
}
