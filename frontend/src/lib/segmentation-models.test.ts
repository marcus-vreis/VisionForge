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
