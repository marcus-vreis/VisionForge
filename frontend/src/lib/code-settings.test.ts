import { describe, expect, it } from "vitest";
import { codeSettingsFor } from "./code-settings";

describe("codeSettingsFor", () => {
  it("reads what the payload decides", () => {
    expect(codeSettingsFor("classification", {
      task: "binary",
      training: { mixed_precision: false },
      transforms: { augment: true },
      model: { name: "resnet18" },
    })).toEqual({ kind: "binary", amp: false, augment: true, model: "resnet18" });
  });

  it("leaves mixed precision undecided unless it is off or the device is the CPU", () => {
    const amp = (device: string) =>
      codeSettingsFor("classification", { training: { mixed_precision: true }, device: { kind: device } }).amp;
    expect(amp("auto")).toBeUndefined();
    expect(amp("cuda")).toBeUndefined();
    expect(amp("cpu")).toBe(false);
  });

  it("ignores values it does not know", () => {
    expect(codeSettingsFor("classification", { task: "regression", model: { name: "" }, transforms: [] })).toEqual({});
  });

  it("reads the loss of a regression run", () => {
    expect(codeSettingsFor("regression", {
      task: "regression",
      model: { name: "resnet18" },
      training: { loss: "huber", mixed_precision: false },
    })).toEqual({ loss: "huber" });
    expect(codeSettingsFor("regression", { training: {} })).toEqual({});
  });

  it("decides nothing for a task without settings", () => {
    expect(codeSettingsFor("custom", { training: { loss: "mse" } })).toEqual({});
  });
});

describe("codeSettingsFor segmentation", () => {
  it("reads the loss and the model the factory builds", () => {
    expect(codeSettingsFor("segmentation", {
      model: { name: "unet", num_classes: 3 },
      training: { loss: "combined" },
    })).toEqual({ loss: "combined", model: "unet" });
  });

  it("leaves the architecture undecided for a model of the researcher's own", () => {
    expect(codeSettingsFor("segmentation", {
      model: { name: "unet", custom_model: "my_net" },
      training: { loss: "dice" },
    })).toEqual({ loss: "dice" });
  });
});

describe("codeSettingsFor anomaly", () => {
  it("reads the method from model.name", () => {
    expect(codeSettingsFor("anomaly", { model: { name: "patchcore", backbone: "resnet18" } })).toEqual({ method: "patchcore" });
    expect(codeSettingsFor("anomaly", { model: { name: "autoencoder" } })).toEqual({ method: "autoencoder" });
    expect(codeSettingsFor("anomaly", { model: { name: "padim" } })).toEqual({});
  });
});

describe("codeSettingsFor detection", () => {
  it("reads model.backend and model.name", () => {
    expect(codeSettingsFor("detection", {
      model: { backend: "torchvision", name: "retinanet_resnet50_fpn" },
    })).toEqual({ backend: "torchvision", model: "retinanet_resnet50_fpn" });
    expect(codeSettingsFor("detection", { model: { backend: "ultralytics", name: "yolo11n" } }))
      .toEqual({ backend: "ultralytics", model: "yolo11n" });
    expect(codeSettingsFor("detection", { model: { backend: "mmdet" } })).toEqual({});
  });
});
