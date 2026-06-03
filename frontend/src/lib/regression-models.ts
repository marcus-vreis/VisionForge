/** Regression model options + form state — mirrors the Python
 *  RegressionModelConfig backbones in visionforge/utils/regression_config.py.
 *  Keep in sync: a name the backend rejects produces a 422 on submit. */

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

/** Controlled form state for a regression run — mirrors RegressionConfig. */
export interface RegressionForm {
  name: string;
  model: {
    name: string;
    num_targets: number;
    pretrained: boolean;
  };
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
  };
}

export function makeDefaultRegressionForm(): RegressionForm {
  return {
    name: "regression_001",
    model: { name: "resnet50", num_targets: 1, pretrained: true },
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
    },
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
      image_size: form.data.image_size,
    },
    training: { ...form.training },
  };
}
