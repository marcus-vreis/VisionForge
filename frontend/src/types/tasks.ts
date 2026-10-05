import type { SelectOption } from "../components/controls";
import type { Dict } from "../i18n/pt";

export interface TaskParam {
  type: "number" | "segmented" | "toggle" | "text";
  key: string;
  label: string;
  hint?: string;
  min?: number;
  max?: number;
  step?: number;
  options?: (string | { value: string; label: string })[];
}

export interface TaskDefinition {
  key: string;
  label: string;
  short: string;
  description: string;
  /** Hex color used for tab indicator dots and overlay progress bar. */
  accent: string;
  models: SelectOption[];
  params: TaskParam[];
  defaults: Record<string, unknown>;
}

/** The five built-in tasks, with their text in the active language. Not a hook,
 *  so the dictionary comes in as an argument: `taskDefinitions(t)` with
 *  `const t = useT()`. Keys, accents, limits and defaults are the same in every
 *  language; a researcher-defined task brings its own text (see lib/custom-tasks.ts). */
export function taskDefinitions(t: Dict): TaskDefinition[] {
  const p = t.tasks.params;
  return [
    {
      key: "classification",
      label: t.tasks.classification.label,
      short: "class",
      description: t.tasks.classification.description,
      accent: "#f16363",
      models: [
        { value: "resnet50",        label: "ResNet-50",        sub: t.tasks.models.resnet50 },
        { value: "resnet18",        label: "ResNet-18",        sub: t.tasks.models.resnet18 },
        { value: "resnet34",        label: "ResNet-34",        sub: t.tasks.models.resnet34 },
        { value: "resnet101",       label: "ResNet-101",       sub: t.tasks.models.resnet101 },
        { value: "efficientnet_b1", label: "EfficientNet-B1",  sub: t.tasks.models.efficientnet_b1 },
        { value: "efficientnet_b7", label: "EfficientNet-B7",  sub: t.tasks.models.efficientnet_b7 },
        { value: "vgg16",           label: "VGG-16",           sub: t.tasks.models.vgg16 },
        { value: "vgg19",           label: "VGG-19",           sub: t.tasks.models.vgg19 },
        { value: "alexnet",         label: "AlexNet",          sub: t.tasks.models.alexnet },
      ],
      defaults: {
        modelo: "resnet50",
        epocas: 50,
        lr: 0.0001,
        batch: 16,
        optimizer: "adam",
        augment: true,
        pretrained: true,
        early_stop: 10,
        seed: 42,
      },
      params: [
        { type: "number",    key: "epocas",     label: p.epochs,        hint: p.epochsHint, min: 1,       max: 5000, step: 1      },
        { type: "number",    key: "lr",         label: p.learningRate, hint: p.learningRateHint, min: 0.000001, max: 1,   step: 0.0001 },
        { type: "number",    key: "batch",      label: p.batchSize,    hint: p.batchSizeHint,  min: 1,       max: 512,  step: 1      },
        {
          type: "segmented",
          key: "optimizer",
          label: p.optimizer,
          options: [
            { value: "adam",  label: "Adam"  },
            { value: "sgd",   label: "SGD"   },
            { value: "adamw", label: "AdamW" },
          ],
        },
        { type: "number",  key: "early_stop", label: p.earlyStop,    hint: p.earlyStopHint, min: 1, max: 200, step: 1 },
        { type: "number",  key: "seed",       label: p.seed,          hint: p.seedHint, min: 0, max: 99999, step: 1 },
        { type: "toggle",  key: "augment",    label: p.augment },
        { type: "toggle",  key: "pretrained", label: p.pretrained },
      ],
    },
    {
      key: "detection",
      label: t.tasks.detection.label,
      short: "detect",
      description: t.tasks.detection.description,
      accent: "#48cf8e",
      models: [
        { value: "yolov8n", label: "YOLOv8-n", sub: t.tasks.models.yolov8n },
        { value: "yolov8s", label: "YOLOv8-s", sub: t.tasks.models.yolov8s },
      ],
      defaults: { modelo: "yolov8s", epocas: 100, lr: 0.01, batch: 16 },
      params: [
        { type: "number", key: "epocas", label: p.epochs, hint: p.epochsHint, min: 1, max: 1000, step: 1 },
        { type: "number", key: "lr",     label: p.learningRate, hint: p.learningRateHint, min: 0.000001, max: 1, step: 0.0001 },
      ],
    },
    {
      key: "regression",
      label: t.tasks.regression.label,
      short: "reg",
      description: t.tasks.regression.description,
      accent: "#5b9fff",
      models: [
        { value: "mlp", label: "MLP (3 layers)", sub: t.tasks.models.mlp },
      ],
      defaults: { modelo: "mlp", epocas: 200, lr: 0.0005, batch: 64 },
      params: [
        { type: "number", key: "epocas", label: p.epochs, hint: p.epochsHint, min: 1, max: 5000, step: 1 },
        { type: "number", key: "lr",     label: p.learningRate, hint: p.learningRateHint, min: 0.000001, max: 1, step: 0.0001 },
      ],
    },
    {
      key: "segmentation",
      label: t.tasks.segmentation.label,
      short: "seg",
      description: t.tasks.segmentation.description,
      accent: "#b079ff",
      models: [
        { value: "unet", label: "U-Net", sub: t.tasks.models.unet },
      ],
      defaults: { modelo: "unet", epocas: 80, lr: 0.0003, batch: 8 },
      params: [
        { type: "number", key: "epocas", label: p.epochs, hint: p.epochsHint, min: 1, max: 1000, step: 1 },
        { type: "number", key: "lr",     label: p.learningRate, hint: p.learningRateHint, min: 0.000001, max: 1, step: 0.0001 },
      ],
    },
    {
      key: "anomaly",
      label: t.tasks.anomaly.label,
      short: "anom",
      description: t.tasks.anomaly.description,
      accent: "#f5a524",
      models: [
        { value: "autoencoder", label: "Autoencoder", sub: t.tasks.models.autoencoder },
        { value: "patchcore", label: "PatchCore", sub: t.tasks.models.patchcore },
      ],
      defaults: { modelo: "autoencoder", epocas: 30, lr: 0.001, batch: 32 },
      params: [
        { type: "number", key: "epocas", label: p.epochs, hint: p.epochsHint, min: 1, max: 1000, step: 1 },
        { type: "number", key: "lr",     label: p.learningRate, hint: p.learningRateHint, min: 0.000001, max: 1, step: 0.0001 },
      ],
    },
  ];
}
