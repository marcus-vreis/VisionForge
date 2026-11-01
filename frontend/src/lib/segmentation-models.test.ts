import { describe, expect, it } from "vitest";
import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import {
  buildSegmentationPayload,
  segmentationFormFromPayload,
  ignoreIndexCollides,
  makeDefaultSegmentationForm,
  segmentationModels,
} from "./segmentation-models";

describe("segmentation-models", () => {
  it("defaults to unet with 2 classes and ignore_index 255", () => {
    const form = makeDefaultSegmentationForm();
    // unet since ADR-100: deeplabv3_resnet50 at 512px exhausts modest GPUs.
    expect(form.model.name).toBe("unet");
    expect(form.model.num_classes).toBe(2);
    expect(form.data.ignore_index).toBe(255);
  });

  it("flags ignore_index collisions with a real class id", () => {
    const form = makeDefaultSegmentationForm();
    form.model.num_classes = 4;
    form.data.ignore_index = 2; // within 0..3 -> collides
    expect(ignoreIndexCollides(form)).toBe(true);
  });

  it("does not flag ignore_index outside the class range", () => {
    const form = makeDefaultSegmentationForm();
    form.model.num_classes = 4;
    form.data.ignore_index = 255;
    expect(ignoreIndexCollides(form)).toBe(false);
    form.data.ignore_index = -1;
    expect(ignoreIndexCollides(form)).toBe(false);
  });

  it("forces num_classes to at least 1 in the payload", () => {
    const form = makeDefaultSegmentationForm();
    form.model.num_classes = 0;
    const payload = buildSegmentationPayload(form);
    const model = payload.model as { num_classes: number };
    expect(model.num_classes).toBe(1);
  });

  it("passes the paired image/mask dataset fields through", () => {
    const form = makeDefaultSegmentationForm();
    form.data.base_dir = "/data/seg";
    const payload = buildSegmentationPayload(form);
    const data = payload.data as { base_dir: string; masks_subdir: string };
    expect(data.base_dir).toBe("/data/seg");
    expect(data.masks_subdir).toBe("masks");
  });

  it("maps transfer-learning mode to the config payload shape", () => {
    const form = makeDefaultSegmentationForm();
    expect(buildSegmentationPayload(form).transfer_learning).toBeNull();

    form.transfer = "feature_extraction";
    expect(buildSegmentationPayload(form).transfer_learning).toEqual({
      mode: "feature_extraction",
    });

    form.transfer = "fine_tuning";
    form.backbone_lr_multiplier = 0.25;
    expect(buildSegmentationPayload(form).transfer_learning).toEqual({
      mode: "fine_tuning",
      backbone_lr_multiplier: 0.25,
    });
  });
});

describe("segmentation-models · transforms & preprocessing (ADR-059)", () => {
  it("sends augmentation flags under data.transforms without image_size", () => {
    const form = makeDefaultSegmentationForm();
    form.transforms.horizontal_flip = false;
    const payload = buildSegmentationPayload(form);
    const data = payload.data as {
      image_size: number;
      transforms: { horizontal_flip: boolean; image_size?: number };
    };
    // resize stays owned by data.image_size (joint image+mask transforms)
    expect(data.image_size).toBe(512);
    expect(data.transforms.image_size).toBeUndefined();
    expect(data.transforms.horizontal_flip).toBe(false);
  });

  it("projects preprocessing steps into schema-flat data.preprocessing.steps", () => {
    const form = makeDefaultSegmentationForm();
    form.preprocessing = [{ kind: "equalize", params: {} }];
    const payload = buildSegmentationPayload(form);
    const data = payload.data as {
      preprocessing: { steps: Array<Record<string, unknown>> };
    };
    expect(data.preprocessing.steps).toEqual([{ kind: "equalize" }]);
  });
});

describe("segmentation-models · YAML round-trip (ADR-059 header)", () => {
  it("formFromPayload(buildPayload(form)) reproduces the form", async () => {
    const { segmentationFormFromPayload } = await import("./segmentation-models");
    const form = makeDefaultSegmentationForm();
    form.name = "seg_rt";
    form.model.name = "unet";
    form.model.num_classes = 5;
    form.data.image_size = 256;
    form.data.ignore_index = 254;
    form.transfer = "feature_extraction";
    form.transforms.color_jitter = true;
    form.preprocessing = [{ kind: "equalize", params: {} }];

    const roundTripped = segmentationFormFromPayload(
      buildSegmentationPayload(form),
    );
    expect(roundTripped).toEqual(form);
  });
});

describe("segmentation-models · dropdown text", () => {
  it("keeps the same models in both languages and words only the sub-line", () => {
    const ptList = segmentationModels(pt);
    const enList = segmentationModels(en);
    expect(enList.map((m) => [m.value, m.label])).toEqual(ptList.map((m) => [m.value, m.label]));
    expect(ptList.every((m) => m.sub)).toBe(true);
    expect(enList.every((m) => m.sub)).toBe(true);
  });

  it("words the sub-line in the active language", () => {
    const sub = (dict: typeof pt, value: string) =>
      segmentationModels(dict).find((m) => m.value === value)?.sub;
    expect(sub(pt, "deeplabv3_mobilenet_v3_large")).toBe("leve · 11M");
    expect(sub(en, "deeplabv3_mobilenet_v3_large")).toBe("light · 11M");
    expect(sub(pt, "unet")).toBe("clássico · 31M");
    expect(sub(en, "unet")).toBe("classic · 31M");
  });
});

describe("YAML import keeps what the form has no field for", () => {
  const imported = {
    name: "recipe",
    model: { name: "deeplabv3_resnet50", num_classes: 2, pretrained: true },
    data: { base_dir: "C:/ds", num_workers: 4, pin_memory: false, image_size: 256 },
    training: {
      epochs: 40,
      optimizer: "adamw",
      weight_decay: 0.01,
      scheduler: { kind: "cosine" },
    },
    output: { models_dir: "C:/out/models" },
    device: { kind: "cuda" },
  };

  it("sends the imported hyperparameters, workers, output and device", () => {
    const payload = buildSegmentationPayload(segmentationFormFromPayload(imported));
    expect(payload.training).toMatchObject({ weight_decay: 0.01, scheduler: { kind: "cosine" } });
    expect(payload.data).toMatchObject({ num_workers: 4, pin_memory: false });
    expect(payload.output).toEqual({ models_dir: "C:/out/models" });
    expect(payload.device).toEqual({ kind: "cuda" });
  });

  it("lets a later edit in the form win over the imported value", () => {
    const form = segmentationFormFromPayload(imported);
    form.training.epochs = 5;
    const payload = buildSegmentationPayload(form);
    expect(payload.training).toMatchObject({ epochs: 5, weight_decay: 0.01 });
  });

  it("a form that never imported sends no extra keys", () => {
    const payload = buildSegmentationPayload(makeDefaultSegmentationForm());
    expect(payload).not.toHaveProperty("output");
    expect(payload.training).not.toHaveProperty("weight_decay");
  });
});
