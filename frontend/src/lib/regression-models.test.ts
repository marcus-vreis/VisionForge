import { describe, expect, it } from "vitest";
import {
  buildRegressionComparePayload,
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

  it("defaults comparison to disabled with two backbones ranked by r2", () => {
    const form = makeDefaultRegressionForm();
    expect(form.compare.enabled).toBe(false);
    expect(form.compare.model_names.length).toBeGreaterThanOrEqual(2);
    expect(form.compare.metric).toBe("r2");
  });

  it("wraps the base config plus model_names/metric in the compare payload", () => {
    const form = makeDefaultRegressionForm();
    form.compare.model_names = ["resnet18", "resnet34"];
    const payload = buildRegressionComparePayload(form);
    expect(payload.model_names).toEqual(["resnet18", "resnet34"]);
    expect(payload.metric).toBe("r2");
    expect((payload.config as { model: { name: string } }).model.name).toBe(
      "resnet50",
    );
  });
});
