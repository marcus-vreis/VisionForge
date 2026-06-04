import { describe, expect, it } from "vitest";
import {
  buildAnomalyPayload,
  isPatchCore,
  makeDefaultAnomalyForm,
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
    expect(model.coreset_ratio).toBe(0.1);
  });
});
