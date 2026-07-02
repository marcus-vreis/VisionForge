import { describe, expect, it } from "vitest";
import {
  buildRegressionPayload,
  makeDefaultRegressionForm,
  parseTargetColumns,
} from "./regression-models";

describe("regression-models", () => {
  it("parses comma-separated target columns, trimming + dropping empties", () => {
    expect(parseTargetColumns("a, b ,, c")).toEqual(["a", "b", "c"]);
    expect(parseTargetColumns("  ")).toEqual([]);
  });

  it("defaults to a single resnet50 target", () => {
    const form = makeDefaultRegressionForm();
    expect(form.model.name).toBe("resnet50");
    expect(form.data.target_columns).toBe("target");
  });

  it("forces num_targets to match the parsed target list", () => {
    const form = makeDefaultRegressionForm();
    form.data.target_columns = "x, y, z";
    const payload = buildRegressionPayload(form);
    const model = payload.model as { num_targets: number };
    const data = payload.data as { target_columns: string[] };
    expect(model.num_targets).toBe(3);
    expect(data.target_columns).toEqual(["x", "y", "z"]);
  });

  it("falls back to a single target when the field is empty", () => {
    const form = makeDefaultRegressionForm();
    form.data.target_columns = "";
    const payload = buildRegressionPayload(form);
    const model = payload.model as { num_targets: number };
    const data = payload.data as { target_columns: string[] };
    expect(model.num_targets).toBe(1);
    expect(data.target_columns).toEqual(["target"]);
  });

  it("sends image size under data.transforms (not the dropped top-level field)", () => {
    const form = makeDefaultRegressionForm();
    form.data.image_size = 384;
    const payload = buildRegressionPayload(form);
    const data = payload.data as {
      image_size?: number;
      transforms: { image_size: number };
    };
    expect(data.transforms.image_size).toBe(384);
    expect(data.image_size).toBeUndefined();
  });

  it("maps transfer-learning mode to the config payload shape", () => {
    const form = makeDefaultRegressionForm();
    expect(buildRegressionPayload(form).transfer_learning).toBeNull();

    form.transfer = "feature_extraction";
    expect(buildRegressionPayload(form).transfer_learning).toEqual({
      mode: "feature_extraction",
    });

    form.transfer = "fine_tuning";
    form.backbone_lr_multiplier = 0.2;
    expect(buildRegressionPayload(form).transfer_learning).toEqual({
      mode: "fine_tuning",
      backbone_lr_multiplier: 0.2,
    });
  });
});

describe("regression-models · transforms & preprocessing (ADR-059 brick A)", () => {
  it("sends the augmentation flags and normalization under data.transforms", () => {
    const form = makeDefaultRegressionForm();
    form.transforms.horizontal_flip = false;
    form.transforms.rotation_degrees = 0;
    form.transforms.color_jitter = true;
    const payload = buildRegressionPayload(form);
    const data = payload.data as {
      transforms: {
        horizontal_flip: boolean;
        rotation_degrees: number;
        color_jitter: boolean;
        normalize_mean: number[];
        normalize_std: number[];
      };
    };
    expect(data.transforms.horizontal_flip).toBe(false);
    expect(data.transforms.rotation_degrees).toBe(0);
    expect(data.transforms.color_jitter).toBe(true);
    expect(data.transforms.normalize_mean).toEqual([0.485, 0.456, 0.406]);
    expect(data.transforms.normalize_std).toEqual([0.229, 0.224, 0.225]);
  });

  it("defaults mirror the backend TransformConfig (behavior-preserving)", () => {
    const form = makeDefaultRegressionForm();
    const payload = buildRegressionPayload(form);
    const data = payload.data as {
      transforms: { horizontal_flip: boolean; rotation_degrees: number };
    };
    expect(data.transforms.horizontal_flip).toBe(true);
    expect(data.transforms.rotation_degrees).toBe(10);
  });

  it("projects UI preprocessing steps into schema-flat data.preprocessing.steps", () => {
    const form = makeDefaultRegressionForm();
    form.preprocessing = [
      { kind: "gaussian_blur", params: { radius: 2 } },
      { kind: "grayscale", params: {} },
    ];
    const payload = buildRegressionPayload(form);
    const data = payload.data as {
      preprocessing: { steps: Array<Record<string, unknown>> };
    };
    expect(data.preprocessing.steps).toEqual([
      { kind: "gaussian_blur", radius: 2 },
      { kind: "grayscale" },
    ]);
  });

  it("falls back to imagenet stats when a normalization triple is malformed", () => {
    const form = makeDefaultRegressionForm();
    form.transforms.normalize_mean = "0.5, abc";
    const payload = buildRegressionPayload(form);
    const data = payload.data as { transforms: { normalize_mean: number[] } };
    expect(data.transforms.normalize_mean).toEqual([0.485, 0.456, 0.406]);
  });
});

describe("regression-models · YAML round-trip (ADR-059 header)", () => {
  it("formFromPayload(buildPayload(form)) reproduces the form", async () => {
    const { regressionFormFromPayload } = await import("./regression-models");
    const form = makeDefaultRegressionForm();
    form.name = "exp_rt";
    form.model.name = "resnet18";
    form.data.target_columns = "x, y";
    form.data.image_size = 384;
    form.training.learning_rate = 0.005;
    form.transfer = "fine_tuning";
    form.backbone_lr_multiplier = 0.2;
    form.transforms.horizontal_flip = false;
    form.transforms.rotation_degrees = 0;
    form.preprocessing = [{ kind: "gaussian_blur", params: { radius: 2 } }];

    const roundTripped = regressionFormFromPayload(buildRegressionPayload(form));
    // num_targets is forced from the parsed target list on export
    const expected = { ...form, model: { ...form.model, num_targets: 2 } };
    expect(roundTripped).toEqual(expected);
  });
});
