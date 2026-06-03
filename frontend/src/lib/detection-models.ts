/** Detection model options per backend — mirrors the Python sets in
 *  visionforge/utils/detection_config.py (_ULTRALYTICS_MODELS / _TORCHVISION_MODELS).
 *  Keep in sync: a name the backend rejects produces a 422 on submit. */

export const DETECTION_BACKENDS = ["ultralytics", "torchvision"] as const;
export type DetectionBackend = (typeof DETECTION_BACKENDS)[number];

export interface DetectionModelOption {
  value: string;
  label: string;
  sub?: string;
}

export const DETECTION_MODELS: Record<DetectionBackend, DetectionModelOption[]> = {
  ultralytics: [
    { value: "yolo11n", label: "YOLO11-n", sub: "nano" },
    { value: "yolo11s", label: "YOLO11-s", sub: "small" },
    { value: "yolo11m", label: "YOLO11-m", sub: "medium" },
    { value: "yolo11l", label: "YOLO11-l", sub: "large" },
    { value: "yolo11x", label: "YOLO11-x", sub: "xlarge" },
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

/** First (recommended) model for a backend — used when switching backends. */
export function defaultModelForBackend(backend: DetectionBackend): string {
  return DETECTION_MODELS[backend][0].value;
}

/** Whether a model name belongs to the given backend. */
export function isValidModelForBackend(
  backend: DetectionBackend,
  name: string,
): boolean {
  return DETECTION_MODELS[backend].some((m) => m.value === name);
}

/** Controlled form state for a detection run — mirrors DetectionConfig. */
export interface DetectionForm {
  name: string;
  model: {
    backend: DetectionBackend;
    name: string;
    num_classes: number;
    pretrained: boolean;
  };
  data: { base_dir: string; image_size: number };
  training: {
    epochs: number;
    batch_size: number;
    learning_rate: number;
    patience: number;
    seed: number;
    workers: number;
  };
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
    data: { base_dir: "", image_size: 640 },
    training: {
      epochs: 100,
      batch_size: 16,
      learning_rate: 0.01,
      patience: 50,
      seed: 0,
      workers: 8,
    },
  };
}
