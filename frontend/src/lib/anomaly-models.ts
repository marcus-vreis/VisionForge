/** Anomaly model options + form state — mirrors the Python AnomalyModelConfig in
 *  visionforge/utils/anomaly_config.py. Keep in sync: a value the backend rejects
 *  produces a 422 on submit. */

import type { PreprocessingStep } from "../components/PreprocessingPanel";
import {
  mergeFormShape,
  stepsFromPayload,
  transformsFormFromPayload,
} from "./form-import";
import {
  buildPreprocessingPayload,
  buildTransformsPayload,
  makeDefaultTransformsForm,
  type TransformsForm,
} from "./transforms-form";

export interface AnomalyModelOption {
  value: string;
  label: string;
  sub?: string;
}

export const ANOMALY_MODELS: AnomalyModelOption[] = [
  { value: "autoencoder", label: "Autoencoder", sub: "reconstrução · treinável" },
  { value: "patchcore", label: "PatchCore", sub: "memory bank · backbone" },
];

export const ANOMALY_BACKBONES: AnomalyModelOption[] = [
  { value: "resnet18", label: "ResNet-18", sub: "light" },
  { value: "resnet34", label: "ResNet-34" },
  { value: "resnet50", label: "ResNet-50" },
  { value: "wide_resnet50_2", label: "Wide-ResNet-50-2", sub: "patchcore padrão" },
];

/** Controlled form state for an anomaly run — mirrors AnomalyConfig. */
export interface AnomalyForm {
  name: string;
  model: {
    name: string;
    backbone: string;
    latent_dim: number;
    coreset_ratio: number;
    pretrained: boolean;
  };
  data: {
    base_dir: string;
    train_dir: string;
    test_dir: string;
    normal_dir: string;
    image_size: number;
  };
  training: {
    epochs: number;
    batch_size: number;
    learning_rate: number;
    optimizer: string;
    early_stopping_patience: number;
    threshold_percentile: number;
    seed: number;
  };
  /** Filter pipeline applied before augmentation (data.preprocessing.steps). */
  preprocessing: PreprocessingStep[];
  /** Augmentation/normalization (data.transforms) — flips/rotations can hurt
   *  orientation-sensitive defects; now visible instead of silently on. */
  transforms: TransformsForm;
}

export function makeDefaultAnomalyForm(): AnomalyForm {
  return {
    name: "anomaly_001",
    model: {
      name: "autoencoder",
      backbone: "resnet18",
      latent_dim: 512,
      coreset_ratio: 0.1,
      pretrained: true,
    },
    data: {
      base_dir: "",
      train_dir: "train",
      test_dir: "test",
      normal_dir: "good",
      image_size: 256,
    },
    training: {
      epochs: 30,
      batch_size: 32,
      learning_rate: 0.001,
      optimizer: "adam",
      early_stopping_patience: 10,
      threshold_percentile: 95,
      seed: 42,
    },
    preprocessing: [],
    transforms: makeDefaultTransformsForm(),
  };
}

/** True when PatchCore is selected (controls which model knobs are relevant). */
export function isPatchCore(form: AnomalyForm): boolean {
  return form.model.name === "patchcore";
}

/** Rebuild the form from an imported AnomalyConfig payload (YAML import).
 *  Inverse of buildAnomalyPayload; unknown/mistyped values fall back to the
 *  defaults instead of corrupting the form. */
export function anomalyFormFromPayload(
  payload: Record<string, unknown>,
): AnomalyForm {
  const form = mergeFormShape(makeDefaultAnomalyForm(), payload);
  const data = (payload.data ?? {}) as Record<string, unknown>;
  form.transforms = transformsFormFromPayload(data.transforms);
  form.preprocessing = stepsFromPayload(data.preprocessing);
  return form;
}

/** Project the form into the AnomalyConfig wire payload. */
export function buildAnomalyPayload(form: AnomalyForm): Record<string, unknown> {
  return {
    name: form.name,
    model: { ...form.model },
    data: {
      ...form.data,
      // Resize is owned by data.image_size for anomaly (synced by the
      // datamodule); transforms carries augmentation flags + normalization.
      transforms: buildTransformsPayload(form.transforms),
      preprocessing: buildPreprocessingPayload(form.preprocessing),
    },
    training: { ...form.training },
  };
}
