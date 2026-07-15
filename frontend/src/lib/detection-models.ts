/** Detection model options per backend — mirrors the Python sets in
 *  visionforge/utils/detection_config.py (_ULTRALYTICS_MODELS / _TORCHVISION_MODELS).
 *  Keep in sync: a name the backend rejects produces a 422 on submit. */

import { mergeFormShape } from "./form-import";

export const DETECTION_BACKENDS = ["ultralytics", "torchvision"] as const;
export type DetectionBackend = (typeof DETECTION_BACKENDS)[number];

export interface DetectionModelOption {
  value: string;
  label: string;
  sub?: string;
}

export const DETECTION_MODELS: Record<DetectionBackend, DetectionModelOption[]> = {
  ultralytics: [
    { value: "yolo26n", label: "YOLO26-n", sub: "nano · NMS-free" },
    { value: "yolo26s", label: "YOLO26-s", sub: "small · NMS-free" },
    { value: "yolo26m", label: "YOLO26-m", sub: "medium · NMS-free" },
    { value: "yolo26l", label: "YOLO26-l", sub: "large · NMS-free" },
    { value: "yolo26x", label: "YOLO26-x", sub: "xlarge · NMS-free" },
    { value: "yolo11n", label: "YOLO11-n", sub: "nano" },
    { value: "yolo11s", label: "YOLO11-s", sub: "small" },
    { value: "yolo11m", label: "YOLO11-m", sub: "medium" },
    { value: "yolo11l", label: "YOLO11-l", sub: "large" },
    { value: "yolo11x", label: "YOLO11-x", sub: "xlarge" },
    { value: "yolo12n", label: "YOLO12-n", sub: "nano" },
    { value: "yolo12s", label: "YOLO12-s", sub: "small" },
    { value: "yolo12m", label: "YOLO12-m", sub: "medium" },
    { value: "yolo12l", label: "YOLO12-l", sub: "large" },
    { value: "yolo12x", label: "YOLO12-x", sub: "xlarge" },
    { value: "yolov10n", label: "YOLOv10-n", sub: "nano · NMS-free" },
    { value: "yolov10s", label: "YOLOv10-s", sub: "small · NMS-free" },
    { value: "yolov10m", label: "YOLOv10-m", sub: "medium · NMS-free" },
    { value: "yolov10b", label: "YOLOv10-b", sub: "balanced · NMS-free" },
    { value: "yolov10l", label: "YOLOv10-l", sub: "large · NMS-free" },
    { value: "yolov10x", label: "YOLOv10-x", sub: "xlarge · NMS-free" },
    { value: "yolov9t", label: "YOLOv9-t", sub: "tiny" },
    { value: "yolov9s", label: "YOLOv9-s", sub: "small" },
    { value: "yolov9m", label: "YOLOv9-m", sub: "medium" },
    { value: "yolov9c", label: "YOLOv9-c", sub: "compact" },
    { value: "yolov9e", label: "YOLOv9-e", sub: "extended" },
    { value: "yolov8n", label: "YOLOv8-n", sub: "nano" },
    { value: "yolov8s", label: "YOLOv8-s", sub: "small" },
    { value: "yolov8m", label: "YOLOv8-m", sub: "medium" },
    { value: "yolov8l", label: "YOLOv8-l", sub: "large" },
    { value: "yolov8x", label: "YOLOv8-x", sub: "xlarge" },
    { value: "rtdetr-l", label: "RT-DETR-l", sub: "transformer" },
    { value: "rtdetr-x", label: "RT-DETR-x", sub: "transformer" },
  ],
  torchvision: [
    { value: "fasterrcnn_resnet50_fpn", label: "Faster R-CNN", sub: "R50-FPN" },
    {
      value: "fasterrcnn_mobilenet_v3_large_fpn",
      label: "Faster R-CNN",
      sub: "MobileNetV3-L-FPN",
    },
    { value: "retinanet_resnet50_fpn", label: "RetinaNet", sub: "R50-FPN" },
    { value: "ssd300_vgg16", label: "SSD300", sub: "VGG16" },
    {
      value: "ssdlite320_mobilenet_v3_large",
      label: "SSDLite320",
      sub: "MobileNetV3-L",
    },
  ],
};

/** Default model for a backend — used when switching backends. Explicit (not
 *  positional) so the list can show newest-first without changing the default,
 *  and so it stays in sync with the Python DetectionModelConfig defaults. */
const DEFAULT_MODEL: Record<DetectionBackend, string> = {
  ultralytics: "yolo11n",
  torchvision: "fasterrcnn_resnet50_fpn",
};

export function defaultModelForBackend(backend: DetectionBackend): string {
  return DEFAULT_MODEL[backend];
}

/** Whether a model name belongs to the given backend. */
export function isValidModelForBackend(
  backend: DetectionBackend,
  name: string,
): boolean {
  return DETECTION_MODELS[backend].some((m) => m.value === name);
}

/** Which dataset source the user provides. ``folder`` synthesizes a data.yaml
 *  from a YOLO-layout root; ``yaml`` points at an existing Ultralytics
 *  data.yaml (e.g. a Roboflow export). Mirrors DetectionDataConfig's
 *  base_dir/data_yaml pair. */
export type DetectionDataSource = "folder" | "yaml";

/** Controlled form state for a detection run — mirrors DetectionConfig. */
export interface DetectionForm {
  name: string;
  model: {
    backend: DetectionBackend;
    name: string;
    num_classes: number;
    pretrained: boolean;
  };
  data: {
    source: DetectionDataSource;
    base_dir: string;
    data_yaml: string;
    image_size: number;
  };
  training: DetectionTrainingForm;
}

export type DetectionOptimizer =
  | "auto"
  | "SGD"
  | "Adam"
  | "Adamax"
  | "AdamW"
  | "NAdam"
  | "RAdam"
  | "RMSProp";

export type DetectionAutoAugment =
  | "randaugment"
  | "autoaugment"
  | "augmix"
  | "none";

/** Augmentation knobs — mirror DetectionAugmentationConfig (Ultralytics). */
export interface DetectionAugmentationForm {
  hsv_h: number;
  hsv_s: number;
  hsv_v: number;
  degrees: number;
  translate: number;
  scale: number;
  shear: number;
  perspective: number;
  flipud: number;
  fliplr: number;
  bgr: number;
  mosaic: number;
  mixup: number;
  copy_paste: number;
  auto_augment: DetectionAutoAugment;
  erasing: number;
}

/** Training knobs — mirror DetectionTrainingConfig (Ultralytics). */
export interface DetectionTrainingForm {
  epochs: number;
  batch_size: number;
  learning_rate: number;
  patience: number;
  seed: number;
  workers: number;
  optimizer: DetectionOptimizer;
  momentum: number;
  weight_decay: number;
  lrf: number;
  cos_lr: boolean;
  warmup_epochs: number;
  warmup_momentum: number;
  warmup_bias_lr: number;
  box: number;
  cls: number;
  dfl: number;
  label_smoothing: number;
  dropout: number;
  nbs: number;
  freeze: number; // first N layers to freeze; 0 = none
  amp: boolean;
  close_mosaic: number;
  single_cls: boolean;
  rect: boolean;
  multi_scale: boolean;
  augmentation: DetectionAugmentationForm;
}

export const DETECTION_OPTIMIZERS: DetectionOptimizer[] = [
  "auto",
  "SGD",
  "Adam",
  "Adamax",
  "AdamW",
  "NAdam",
  "RAdam",
  "RMSProp",
];

/** Project the training form into the wire payload. ``auto_augment: "none"`` is
 *  a UI sentinel for the backend's ``null`` (off); freeze stays nullable. */
export function buildDetectionTrainingPayload(
  t: DetectionTrainingForm,
): Record<string, unknown> {
  const { augmentation, ...rest } = t;
  const { auto_augment, ...aug } = augmentation;
  return {
    ...rest,
    augmentation: {
      ...aug,
      auto_augment: auto_augment === "none" ? null : auto_augment,
    },
  };
}

/** Project the form's dataset section into the wire payload, sending only the
 *  active source. An empty `base_dir` resolves to "." on the server (a valid
 *  dir) and produces a confusing "missing splits" error, so we omit the
 *  inactive field entirely and let DetectionDataConfig validate one source. */
export function buildDetectionDataPayload(
  data: DetectionForm["data"],
): Record<string, unknown> {
  const common = { image_size: data.image_size };
  return data.source === "yaml"
    ? { ...common, data_yaml: data.data_yaml }
    : { ...common, base_dir: data.base_dir };
}

/** Rebuild the form from an imported DetectionConfig payload (YAML import).
 *  Inverse of the buildDetection*Payload pair: the dataset source is derived
 *  from which field the payload carries, the backend `null` auto_augment maps
 *  back to the "none" UI sentinel, and an invalid backend/model pair falls
 *  back to the backend's default model. */
export function detectionFormFromPayload(
  payload: Record<string, unknown>,
): DetectionForm {
  const form = mergeFormShape(makeDefaultDetectionForm(), payload);
  const data = (payload.data ?? {}) as Record<string, unknown>;
  const training = (payload.training ?? {}) as Record<string, unknown>;
  const aug = (training.augmentation ?? {}) as Record<string, unknown>;

  form.data.source = typeof data.data_yaml === "string" && data.data_yaml !== ""
    ? "yaml"
    : "folder";
  if (aug.auto_augment === null) {
    form.training.augmentation.auto_augment = "none";
  }
  if (!isValidModelForBackend(form.model.backend, form.model.name)) {
    form.model.name = defaultModelForBackend(form.model.backend);
  }
  return form;
}

export function makeDefaultDetectionForm(): DetectionForm {
  return {
    name: "detection_001",
    model: {
      backend: "ultralytics",
      name: "yolo11n",
      num_classes: 1,
      pretrained: true,
    },
    data: { source: "folder", base_dir: "", data_yaml: "", image_size: 640 },
    training: makeDefaultDetectionTraining(),
  };
}

/** Ultralytics-faithful training defaults — must match DetectionTrainingConfig
 *  and DetectionAugmentationConfig so an unmodified form is behaviour-preserving. */
export function makeDefaultDetectionTraining(): DetectionTrainingForm {
  return {
    epochs: 100,
    batch_size: 16,
    learning_rate: 0.01,
    patience: 50,
    seed: 0,
    // 2, not Ultralytics' 8: on Windows every DataLoader worker is a spawned
    // process reloading torch's CUDA DLLs — 8 exhausts the page file
    // (WinError 1455). Linux users can raise it for throughput.
    workers: 2,
    optimizer: "auto",
    momentum: 0.937,
    weight_decay: 0.0005,
    lrf: 0.01,
    cos_lr: false,
    warmup_epochs: 3.0,
    warmup_momentum: 0.8,
    warmup_bias_lr: 0.1,
    box: 7.5,
    cls: 0.5,
    dfl: 1.5,
    label_smoothing: 0.0,
    dropout: 0.0,
    nbs: 64,
    freeze: 0,
    amp: true,
    close_mosaic: 10,
    single_cls: false,
    rect: false,
    multi_scale: false,
    augmentation: {
      hsv_h: 0.015,
      hsv_s: 0.7,
      hsv_v: 0.4,
      degrees: 0.0,
      translate: 0.1,
      scale: 0.5,
      shear: 0.0,
      perspective: 0.0,
      flipud: 0.0,
      fliplr: 0.5,
      bgr: 0.0,
      mosaic: 1.0,
      mixup: 0.0,
      copy_paste: 0.0,
      auto_augment: "randaugment",
      erasing: 0.4,
    },
  };
}
