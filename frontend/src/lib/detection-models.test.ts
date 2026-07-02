import { describe, expect, it } from "vitest";
import {
  DETECTION_MODELS,
  buildDetectionDataPayload,
  buildDetectionTrainingPayload,
  defaultModelForBackend,
  isValidModelForBackend,
  makeDefaultDetectionForm,
  makeDefaultDetectionTraining,
} from "./detection-models";

describe("detection-models", () => {
  it("defaults to the first model of each backend", () => {
    expect(defaultModelForBackend("ultralytics")).toBe("yolo11n");
    expect(defaultModelForBackend("torchvision")).toBe("fasterrcnn_resnet50_fpn");
  });

  it("validates membership per backend", () => {
    expect(isValidModelForBackend("ultralytics", "yolov8s")).toBe(true);
    expect(isValidModelForBackend("ultralytics", "fasterrcnn_resnet50_fpn")).toBe(
      false,
    );
    expect(isValidModelForBackend("torchvision", "ssd300_vgg16")).toBe(true);
    expect(isValidModelForBackend("torchvision", "yolo11n")).toBe(false);
  });

  it("has no model shared across backends", () => {
    const u = new Set(DETECTION_MODELS.ultralytics.map((m) => m.value));
    const overlap = DETECTION_MODELS.torchvision.filter((m) => u.has(m.value));
    expect(overlap).toEqual([]);
  });

  it("offers every YOLO family including YOLO26", () => {
    const u = DETECTION_MODELS.ultralytics.map((m) => m.value);
    for (const name of [
      "yolov8n",
      "yolov9c",
      "yolov10b",
      "yolo11n",
      "yolo12m",
      "yolo26n",
      "yolo26x",
      "rtdetr-l",
    ]) {
      expect(u).toContain(name);
    }
  });

  it("keeps yolo11n as the explicit ultralytics default despite list order", () => {
    expect(DETECTION_MODELS.ultralytics[0].value).toBe("yolo26n");
    expect(defaultModelForBackend("ultralytics")).toBe("yolo11n");
  });

  it("maps the auto_augment 'none' sentinel to null in the payload", () => {
    const t = makeDefaultDetectionTraining();
    t.augmentation.auto_augment = "none";
    const payload = buildDetectionTrainingPayload(t) as {
      augmentation: { auto_augment: unknown };
    };
    expect(payload.augmentation.auto_augment).toBeNull();
  });

  it("forwards a real auto_augment value unchanged", () => {
    const payload = buildDetectionTrainingPayload(
      makeDefaultDetectionTraining(),
    ) as { augmentation: { auto_augment: unknown }; optimizer: string };
    expect(payload.augmentation.auto_augment).toBe("randaugment");
    expect(payload.optimizer).toBe("auto");
  });

  it("defaults to the folder dataset source", () => {
    const form = makeDefaultDetectionForm();
    expect(form.data.source).toBe("folder");
    expect(form.data.data_yaml).toBe("");
  });

  it("sends only base_dir in folder mode", () => {
    const payload = buildDetectionDataPayload({
      source: "folder",
      base_dir: "/data/yolo",
      data_yaml: "/ignored.yaml",
      image_size: 640,
    });
    expect(payload).toEqual({ base_dir: "/data/yolo", image_size: 640 });
    expect(payload).not.toHaveProperty("data_yaml");
  });

  it("sends only data_yaml in yaml mode", () => {
    const payload = buildDetectionDataPayload({
      source: "yaml",
      base_dir: "/ignored",
      data_yaml: "/data/data.yaml",
      image_size: 512,
    });
    expect(payload).toEqual({ data_yaml: "/data/data.yaml", image_size: 512 });
    expect(payload).not.toHaveProperty("base_dir");
  });
});

describe("detection-models · YAML round-trip (ADR-059 header)", () => {
  it("formFromPayload(export payload) reproduces the form", async () => {
    const {
      detectionFormFromPayload,
      makeDefaultDetectionForm,
      buildDetectionDataPayload,
      buildDetectionTrainingPayload,
    } = await import("./detection-models");
    const form = makeDefaultDetectionForm();
    form.name = "det_rt";
    form.model.num_classes = 3;
    form.data.source = "yaml";
    form.data.data_yaml = "C:/ds/data.yaml";
    form.training.epochs = 25;
    form.training.optimizer = "AdamW";
    form.training.augmentation.mosaic = 0.5;
    form.training.augmentation.auto_augment = "none";

    const payload = {
      name: form.name,
      model: { ...form.model },
      data: buildDetectionDataPayload(form.data),
      training: buildDetectionTrainingPayload(form.training),
    };
    const roundTripped = detectionFormFromPayload(payload);
    expect(roundTripped).toEqual(form);
  });

  it("derives source=folder and falls back on an invalid backend/model pair", async () => {
    const { detectionFormFromPayload } = await import("./detection-models");
    const form = detectionFormFromPayload({
      name: "x",
      model: { backend: "torchvision", name: "yolo11n" },
      data: { base_dir: "C:/ds", image_size: 640 },
    });
    expect(form.data.source).toBe("folder");
    expect(form.model.name).toBe("fasterrcnn_resnet50_fpn");
  });
});
