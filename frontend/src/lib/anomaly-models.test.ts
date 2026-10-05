import { describe, expect, it } from "vitest";
import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import {
  buildAnomalyPayload,
  isPatchCore,
  makeDefaultAnomalyForm,
  anomalyBackbones,
  anomalyModels,
} from "./anomaly-models";

describe("anomaly-models", () => {
  it("defaults to an autoencoder with percentile 95", () => {
    const form = makeDefaultAnomalyForm();
    expect(form.model.name).toBe("autoencoder");
    expect(form.training.threshold_percentile).toBe(95);
    expect(form.data.normal_dir).toBe("good");
  });

  it("detects PatchCore selection", () => {
    const form = makeDefaultAnomalyForm();
    expect(isPatchCore(form)).toBe(false);
    form.model.name = "patchcore";
    expect(isPatchCore(form)).toBe(true);
  });

  it("passes model, data and training through to the payload", () => {
    const form = makeDefaultAnomalyForm();
    form.data.base_dir = "/data/mvtec/bottle";
    const payload = buildAnomalyPayload(form);
    const data = payload.data as { base_dir: string; normal_dir: string };
    const model = payload.model as { name: string; coreset_ratio: number };
    expect(data.base_dir).toBe("/data/mvtec/bottle");
    expect(data.normal_dir).toBe("good");
    expect(model.name).toBe("autoencoder");
    expect(model.coreset_ratio).toBe(0.01);
  });
});

describe("anomaly-models · transforms & preprocessing (ADR-059)", () => {
  it("exposes the previously-silent augmentation defaults in the payload", () => {
    const form = makeDefaultAnomalyForm();
    const payload = buildAnomalyPayload(form);
    const data = payload.data as {
      transforms: { horizontal_flip: boolean; rotation_degrees: number };
    };
    expect(data.transforms.horizontal_flip).toBe(true);
    expect(data.transforms.rotation_degrees).toBe(10);
  });

  it("lets the user disable flips for orientation-sensitive defects", () => {
    const form = makeDefaultAnomalyForm();
    form.transforms.horizontal_flip = false;
    form.transforms.rotation_degrees = 0;
    const payload = buildAnomalyPayload(form);
    const data = payload.data as {
      transforms: { horizontal_flip: boolean; rotation_degrees: number };
    };
    expect(data.transforms.horizontal_flip).toBe(false);
    expect(data.transforms.rotation_degrees).toBe(0);
  });

  it("projects preprocessing steps into schema-flat data.preprocessing.steps", () => {
    const form = makeDefaultAnomalyForm();
    form.preprocessing = [{ kind: "median_blur", params: { size: 3 } }];
    const payload = buildAnomalyPayload(form);
    const data = payload.data as {
      preprocessing: { steps: Array<Record<string, unknown>> };
    };
    expect(data.preprocessing.steps).toEqual([{ kind: "median_blur", size: 3 }]);
  });
});

describe("anomaly-models · YAML round-trip (ADR-059 header)", () => {
  it("formFromPayload(buildPayload(form)) reproduces the form", async () => {
    const { anomalyFormFromPayload } = await import("./anomaly-models");
    const form = makeDefaultAnomalyForm();
    form.name = "anom_rt";
    form.model.name = "patchcore";
    form.model.backbone = "wide_resnet50_2";
    form.model.coreset_ratio = 0.25;
    form.training.threshold_percentile = 99;
    form.transforms.horizontal_flip = false;
    form.preprocessing = [{ kind: "median_blur", params: { size: 3 } }];

    const roundTripped = anomalyFormFromPayload(buildAnomalyPayload(form));
    expect(roundTripped).toEqual(form);
  });
});

describe("anomaly-models · dropdown text", () => {
  it("keeps the same methods and backbones in both languages", () => {
    for (const make of [anomalyModels, anomalyBackbones]) {
      expect(make(en).map((m) => [m.value, m.label])).toEqual(
        make(pt).map((m) => [m.value, m.label]),
      );
    }
  });

  it("words the sub-line in the active language", () => {
    const method = (dict: typeof pt, value: string) =>
      anomalyModels(dict).find((m) => m.value === value)?.sub;
    expect(method(pt, "autoencoder")).toBe("reconstrução · treinável");
    expect(method(en, "autoencoder")).toBe("reconstruction · trainable");
    expect(method(en, "patchcore")).toBe("memory bank · backbone");

    const backbone = (dict: typeof pt, value: string) =>
      anomalyBackbones(dict).find((m) => m.value === value)?.sub;
    expect(backbone(pt, "wide_resnet50_2")).toBe("patchcore padrão");
    expect(backbone(en, "wide_resnet50_2")).toBe("patchcore default");
    // ResNet-34 and ResNet-50 carry no note, in either language.
    expect(backbone(pt, "resnet34")).toBeUndefined();
    expect(backbone(en, "resnet50")).toBeUndefined();
  });
});