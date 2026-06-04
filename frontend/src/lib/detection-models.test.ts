import { describe, expect, it } from "vitest";
import {
  DETECTION_MODELS,
  buildDetectionDataPayload,
  defaultModelForBackend,
  isValidModelForBackend,
  makeDefaultDetectionForm,
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
