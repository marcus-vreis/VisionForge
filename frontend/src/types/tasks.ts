import type { SelectOption } from "../components/controls";

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

export const TASKS: TaskDefinition[] = [
  {
    key: "classification",
    label: "Classificação",
    short: "class",
    description: "Categorize imagens em rótulos discretos",
    accent: "#f16363",
    models: [
      { value: "resnet50",        label: "ResNet-50",        sub: "imagenet · 25.6M params" },
      { value: "resnet18",        label: "ResNet-18",        sub: "light · 11.7M params" },
      { value: "resnet34",        label: "ResNet-34",        sub: "imagenet · 21.8M params" },
      { value: "resnet101",       label: "ResNet-101",       sub: "deep · 44.5M params" },
      { value: "efficientnet_b1", label: "EfficientNet-B1",  sub: "eficiente · 7.8M params" },
      { value: "efficientnet_b7", label: "EfficientNet-B7",  sub: "grande · 66M params" },
      { value: "vgg16",           label: "VGG-16",           sub: "clássico · 138M params" },
      { value: "vgg19",           label: "VGG-19",           sub: "clássico · 144M params" },
      { value: "alexnet",         label: "AlexNet",          sub: "baseline · 61M params" },
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
      { type: "number",    key: "epocas",     label: "Épocas",        hint: "iterações", min: 1,       max: 5000, step: 1      },
      { type: "number",    key: "lr",         label: "Learning Rate", hint: "lr",        min: 0.000001, max: 1,   step: 0.0001 },
      { type: "number",    key: "batch",      label: "Batch size",    hint: "amostras",  min: 1,       max: 512,  step: 1      },
      {
        type: "segmented",
        key: "optimizer",
        label: "Otimizador",
        options: [
          { value: "adam",  label: "Adam"  },
          { value: "sgd",   label: "SGD"   },
          { value: "adamw", label: "AdamW" },
        ],
      },
      { type: "number",  key: "early_stop", label: "Early stop",    hint: "paciência", min: 1, max: 200, step: 1 },
      { type: "number",  key: "seed",       label: "Seed",          hint: "reprodução", min: 0, max: 99999, step: 1 },
      { type: "toggle",  key: "augment",    label: "Data augmentation" },
      { type: "toggle",  key: "pretrained", label: "Pesos pré-treinados" },
    ],
  },
  {
    key: "detection",
    label: "Detecção de Objeto",
    short: "detect",
    description: "Localize e classifique objetos com bounding boxes",
    accent: "#48cf8e",
    models: [
      { value: "yolov8n", label: "YOLOv8-n", sub: "nano · 3.2M" },
      { value: "yolov8s", label: "YOLOv8-s", sub: "small · 11.2M" },
    ],
    defaults: { modelo: "yolov8s", epocas: 100, lr: 0.01, batch: 16 },
    params: [
      { type: "number", key: "epocas", label: "Épocas", hint: "iterações", min: 1, max: 1000, step: 1 },
      { type: "number", key: "lr",     label: "Learning Rate", hint: "lr", min: 0.000001, max: 1, step: 0.0001 },
    ],
  },
  {
    key: "regression",
    label: "Regressão",
    short: "reg",
    description: "Estime valores contínuos a partir das entradas",
    accent: "#5b9fff",
    models: [
      { value: "mlp", label: "MLP (3 layers)", sub: "baseline" },
    ],
    defaults: { modelo: "mlp", epocas: 200, lr: 0.0005, batch: 64 },
    params: [
      { type: "number", key: "epocas", label: "Épocas", hint: "iterações", min: 1, max: 5000, step: 1 },
      { type: "number", key: "lr",     label: "Learning Rate", hint: "lr", min: 0.000001, max: 1, step: 0.0001 },
    ],
  },
  {
    key: "segmentation",
    label: "Segmentação",
    short: "seg",
    description: "Máscaras a nível de pixel para cada categoria",
    accent: "#b079ff",
    models: [
      { value: "unet", label: "U-Net", sub: "clássico · 31M params" },
    ],
    defaults: { modelo: "unet", epocas: 80, lr: 0.0003, batch: 8 },
    params: [
      { type: "number", key: "epocas", label: "Épocas", hint: "iterações", min: 1, max: 1000, step: 1 },
      { type: "number", key: "lr",     label: "Learning Rate", hint: "lr", min: 0.000001, max: 1, step: 0.0001 },
    ],
  },
];
