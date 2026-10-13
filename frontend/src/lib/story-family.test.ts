import { describe, expect, it } from "vitest";

import { pt } from "../i18n/pt";
import { anomalyModels } from "./anomaly-models";
import { DETECTION_BACKENDS, DETECTION_MODELS } from "./detection-models";
import { regressionModels } from "./regression-models";
import { segmentationModels } from "./segmentation-models";
import { familyForModel, GENERIC_STORY, storySpecFor, type StoryFamily } from "./story-family";

/** The architectures the classification form offers (utils/config.py's
 *  `_ArchitectureLiteral`), which are the ones the regression form offers too. */
const CLASSIFICATION_MODELS = regressionModels(pt).map((m) => m.value);

describe("the family of every model the interface offers", () => {
  it("draws every built-in classification and regression backbone as a CNN, except the attention ones", () => {
    expect(CLASSIFICATION_MODELS).toHaveLength(12);
    const expected: Record<string, StoryFamily> = {
      resnet18: "cnn",
      resnet34: "cnn",
      resnet50: "cnn",
      resnet101: "cnn",
      efficientnet_b1: "cnn",
      efficientnet_b7: "cnn",
      vgg16: "cnn",
      vgg19: "cnn",
      alexnet: "cnn",
      convnext_tiny: "cnn",
      vit_b_16: "transformer",
      swin_t: "transformer",
    };
    expect(Object.keys(expected).sort()).toEqual([...CLASSIFICATION_MODELS].sort());
    for (const name of CLASSIFICATION_MODELS) {
      for (const task of ["classification", "regression"] as const) {
        expect(storySpecFor(task, { model: { name } }).family, `${task} ${name}`).toBe(expected[name]);
      }
    }
  });

  it("draws every detector, Ultralytics and torchvision, as a detector", () => {
    let seen = 0;
    for (const backend of DETECTION_BACKENDS) {
      for (const m of DETECTION_MODELS[backend]) {
        seen += 1;
        expect(storySpecFor("detection", { model: { backend, name: m.value } }).family, m.value).toBe(
          "detector",
        );
      }
    }
    // The transformer detector is a detector: the drawing is backbone, neck, heads.
    expect(DETECTION_MODELS.ultralytics.some((m) => m.value.startsWith("rtdetr"))).toBe(true);
    expect(seen).toBeGreaterThan(30);
  });

  it("draws every segmentation model as an encoder-decoder", () => {
    const names = segmentationModels(pt).map((m) => m.value);
    expect(names).toContain("unet");
    expect(names).toContain("lraspp_mobilenet_v3_large");
    for (const name of names) {
      expect(storySpecFor("segmentation", { model: { name } }).family, name).toBe("encoder_decoder");
    }
  });

  it("tells the autoencoder from PatchCore", () => {
    const names = anomalyModels(pt).map((m) => m.value);
    expect(names.sort()).toEqual(["autoencoder", "patchcore"]);
    expect(storySpecFor("anomaly", { model: { name: "autoencoder" } }).family).toBe("autoencoder");
    expect(storySpecFor("anomaly", { model: { name: "patchcore" } }).family).toBe("patchcore");
  });
});

describe("storySpecFor", () => {
  it("makes regression a network with a one-neuron head, and the rest class scores", () => {
    expect(storySpecFor("regression", { model: { name: "resnet18" } })).toEqual({
      task: "regression",
      family: "cnn",
      output: "value",
      metric: "r2",
    });
    expect(storySpecFor("classification", { model: { name: "resnet18" } }).output).toBe("classes");
    // A regression on a transformer keeps its head: the family is the backbone's.
    expect(storySpecFor("regression", { model: { name: "swin_t" } })).toMatchObject({
      family: "transformer",
      output: "value",
    });
  });

  it("gives each task its headline metric", () => {
    const metric = (task: string) => storySpecFor(task, {}).metric;
    expect(metric("classification")).toBe("accuracy");
    expect(metric("detection")).toBe("map50");
    expect(metric("regression")).toBe("r2");
    expect(metric("segmentation")).toBe("miou");
    expect(metric("anomaly")).toBe("auroc");
    expect(metric("custom")).toBe("primary");
  });

  it("files classification's problem types (binary, multiclass) under classification", () => {
    expect(storySpecFor("binary", { model: { name: "vit_b_16" } })).toMatchObject({
      task: "classification",
      family: "transformer",
    });
  });

  it("gives a researcher's own task the generic network", () => {
    expect(storySpecFor("custom", {})).toEqual(GENERIC_STORY);
    expect(storySpecFor("custom:tarefa", { model: { name: "vit_b_16" } })).toEqual(GENERIC_STORY);
  });

  it("gives a model of the researcher's own (custom_model) the generic network", () => {
    expect(
      storySpecFor("classification", { model: { name: "vit_b_16", custom_model: "minha_rede" } })
        .family,
    ).toBe("cnn");
  });

  it("reads a timm model by its name", () => {
    const timm = (name: string) =>
      storySpecFor("classification", { model: { name: "resnet18", timm_model: name } }).family;
    for (const name of [
      "vit_base_patch16_224",
      "deit_small_patch16_224",
      "swin_tiny_patch4_window7_224",
      "maxvit_tiny_tf_224",
      "mobilevit_s",
      "beit_base_patch16_224",
      "eva02_base_patch14_224",
    ]) {
      expect(timm(name), name).toBe("transformer");
    }
    for (const name of ["resnet50d", "efficientnet_b3", "convnext_small", "regnety_016", "mobilenetv3_large_100"]) {
      expect(timm(name), name).toBe("cnn");
    }
  });

  it("draws a model comparison as the family its models share, else the generic one", () => {
    const comparison = (...names: string[]) =>
      storySpecFor("classification", {
        block: "model_comparison",
        model: { name: "resnet18" },
        model_comparison: { model_names: names },
      }).family;
    expect(comparison("resnet18", "vgg16")).toBe("cnn");
    expect(comparison("vit_b_16", "swin_t")).toBe("transformer");
    expect(comparison("resnet18", "vit_b_16")).toBe("cnn");
  });

  it("ignores a comparison list left in the form by another block", () => {
    expect(
      storySpecFor("classification", {
        block: "classification",
        model: { name: "vit_b_16" },
        model_comparison: { model_names: ["resnet18", "resnet50"] },
      }).family,
    ).toBe("transformer");
  });

  it("falls back to a CNN for a config that says nothing", () => {
    expect(storySpecFor("classification", null).family).toBe("cnn");
    expect(storySpecFor("classification", {}).family).toBe("cnn");
    expect(storySpecFor("classification", { model: "resnet18" }).family).toBe("cnn");
    expect(storySpecFor("classification", { model: { name: "" } }).family).toBe("cnn");
    expect(familyForModel("classification", undefined)).toBe("cnn");
  });
});
