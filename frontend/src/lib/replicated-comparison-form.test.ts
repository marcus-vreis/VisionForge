import { describe, expect, it } from "vitest";
import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import {
  BASELINE_LABEL,
  buildReplicatedComparisonPayload,
  comparisonSeedCount,
  comparisonVariantCount,
  parseAxisValues,
  powerFloor,
  variantLabel,
  wilcoxonFloor,
  type ReplicatedComparisonFields,
} from "./replicated-comparison-form";

const fields: ReplicatedComparisonFields = {
  path: "training.learning_rate",
  rawValues: "0.0001, 0.01",
  seedMode: "auto",
  nReplicates: 6,
  rawSeeds: "42, 43",
  metric: "r2",
};

function problemOf(overrides: Partial<ReplicatedComparisonFields>, dict = pt): string | null {
  const built = buildReplicatedComparisonPayload(dict, { ...fields, ...overrides });
  return "problem" in built ? built.problem : null;
}

describe("buildReplicatedComparisonPayload (ADR-193)", () => {
  it("sends the form as a baseline with no overrides and one variant per value", () => {
    const built = buildReplicatedComparisonPayload(pt, fields);

    expect(built).toEqual({
      payload: {
        variants: {
          [BASELINE_LABEL]: {},
          "learning_rate=0.0001": { "training.learning_rate": 0.0001 },
          "learning_rate=0.01": { "training.learning_rate": 0.01 },
        },
        metric: "r2",
        n_replicates: 6,
      },
    });
  });

  it("sends explicit seeds instead of a count", () => {
    const built = buildReplicatedComparisonPayload(pt, {
      ...fields,
      seedMode: "explicit",
      rawSeeds: "7, 8, 9",
    });

    expect("payload" in built && built.payload.seeds).toEqual([7, 8, 9]);
    expect("payload" in built && "n_replicates" in built.payload).toBe(false);
  });

  it("keeps a non-numeric value as text", () => {
    const built = buildReplicatedComparisonPayload(pt, {
      ...fields,
      path: " model.name ",
      rawValues: "resnet50",
    });

    expect("payload" in built && built.payload.variants).toEqual({
      baseline: {},
      "name=resnet50": { "model.name": "resnet50" },
    });
  });

  it("refuses fewer than 2 variants: the form alone compares with nothing", () => {
    expect(problemOf({ rawValues: "" })).toBe(pt.replicatedComparisonForm.needValue);
    expect(problemOf({ rawValues: " , " }, en)).toBe(en.replicatedComparisonForm.needValue);
  });

  it("refuses more than 4 variants", () => {
    expect(problemOf({ rawValues: "1, 2, 3" })).toBeNull();
    expect(problemOf({ rawValues: "1, 2, 3, 4" })).toBe(pt.replicatedComparisonForm.tooManyValues);
  });

  it("refuses duplicate seeds, as the backend's 422 does", () => {
    expect(problemOf({ seedMode: "explicit", rawSeeds: "1, 1, 2" })).toBe(
      pt.replicatesForm.duplicateSeeds,
    );
    expect(problemOf({ seedMode: "explicit", rawSeeds: "1" })).toBe(
      pt.replicatesForm.needTwoSeeds,
    );
  });

  it("caps an explicit list at 20 seeds, the backend's n_replicates cap", () => {
    const twenty = Array.from({ length: 20 }, (_, i) => i).join(", ");
    const twentyOne = Array.from({ length: 21 }, (_, i) => i).join(", ");

    expect(problemOf({ seedMode: "explicit", rawSeeds: twenty })).toBeNull();
    expect(problemOf({ seedMode: "explicit", rawSeeds: twentyOne })).toBe(
      pt.replicatedComparisonForm.tooManySeeds(20),
    );
  });

  it("refuses a missing field, an output path and repeated values", () => {
    expect(problemOf({ path: "  " })).toBe(pt.replicatedComparisonForm.needPath);
    expect(problemOf({ path: "output.models_dir" })).toBe(pt.replicatedComparisonForm.outputPath);
    // 0.01 and 1e-2 are the same number, so the same variant twice.
    expect(problemOf({ rawValues: "0.01, 1e-2" })).toBe(
      pt.replicatedComparisonForm.duplicateValues,
    );
  });
});

describe("replicated comparison helpers", () => {
  it("names a variant with characters every file system takes", () => {
    expect(variantLabel("data.root", "C:/data sets")).toBe("root=C-data-sets");
  });

  it("parses values as numbers when they are numbers", () => {
    expect(parseAxisValues("0.001, adamw, ,3")).toEqual([0.001, "adamw", 3]);
  });

  it("counts variants and seeds as the cost shows them", () => {
    expect(comparisonVariantCount("0.1, 0.2")).toBe(3);
    expect(comparisonVariantCount("")).toBe(1);
    expect(comparisonSeedCount({ seedMode: "auto", nReplicates: 5, rawSeeds: "1, 2" })).toBe(5);
    expect(comparisonSeedCount({ seedMode: "explicit", nReplicates: 5, rawSeeds: "1, 2" })).toBe(2);
  });

  it("gives Wilcoxon's floor 2/2^n, the backend's min_achievable_p", () => {
    expect(wilcoxonFloor(5)).toBe(0.0625);
    expect(wilcoxonFloor(6)).toBe(0.03125);
    expect(wilcoxonFloor(0)).toBe(1);
  });

  it("warns below 6 seeds at alpha 0.05 and not from 6 on", () => {
    expect(powerFloor(5)).toBe(0.0625);
    expect(powerFloor(2)).toBe(0.5);
    expect(powerFloor(6)).toBeNull();
    expect(powerFloor(20)).toBeNull();
  });
});
