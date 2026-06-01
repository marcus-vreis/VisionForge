import { describe, expect, it } from "vitest";
import {
  DETECTION_MODELS,
  defaultModelForBackend,
  isValidModelForBackend,
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
});
