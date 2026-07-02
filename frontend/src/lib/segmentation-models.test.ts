import { describe, expect, it } from "vitest";
import {
  buildSegmentationPayload,
  ignoreIndexCollides,
  makeDefaultSegmentationForm,
} from "./segmentation-models";

describe("segmentation-models", () => {
  it("defaults to deeplabv3_resnet50 with 2 classes and ignore_index 255", () => {
    const form = makeDefaultSegmentationForm();
    expect(form.model.name).toBe("deeplabv3_resnet50");
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

describe("segmentation-models · transforms & preprocessing (ADR-059 brick A)", () => {
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
