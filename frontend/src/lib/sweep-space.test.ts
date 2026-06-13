import { describe, expect, it } from "vitest";
import {
  buildSearchSpace,
  gridTrialCount,
  makeSweepRow,
  type SweepRow,
} from "./sweep-space";

function row(partial: Partial<SweepRow>): SweepRow {
  return { ...makeSweepRow(), ...partial };
}

describe("buildSearchSpace — grid", () => {
  it("parses comma-separated values, coercing numbers", () => {
    const space = buildSearchSpace("grid", [
      row({ path: "training.learning_rate", values: "0.001, 0.01, 0.1" }),
    ]);
    expect(space).toEqual({ "training.learning_rate": [0.001, 0.01, 0.1] });
  });

  it("keeps non-numeric tokens as strings and skips empty paths", () => {
    const space = buildSearchSpace("grid", [
      row({ path: "model.name", values: "resnet18, resnet50" }),
      row({ path: "", values: "1, 2" }),
    ]);
    expect(space).toEqual({ "model.name": ["resnet18", "resnet50"] });
  });
});

describe("buildSearchSpace — random", () => {
  it("builds a uniform spec from low/high", () => {
    const space = buildSearchSpace("random", [
      row({ path: "training.learning_rate", kind: "uniform", low: "0.001", high: "0.1" }),
    ]);
    expect(space).toEqual({
      "training.learning_rate": { type: "uniform", low: 0.001, high: 0.1 },
    });
  });

  it("builds a choice spec from options", () => {
    const space = buildSearchSpace("random", [
      row({ path: "model.name", kind: "choice", options: "resnet18, resnet50" }),
    ]);
    expect(space).toEqual({
      "model.name": { type: "choice", options: ["resnet18", "resnet50"] },
    });
  });

  it("skips a uniform row with a missing bound", () => {
    const space = buildSearchSpace("random", [
      row({ path: "training.learning_rate", kind: "uniform", low: "0.001", high: "" }),
    ]);
    expect(space).toEqual({});
  });
});

describe("gridTrialCount", () => {
  it("multiplies value-list lengths (cartesian product)", () => {
    expect(
      gridTrialCount([
        row({ path: "training.learning_rate", values: "0.01, 0.1" }),
        row({ path: "training.batch_size", values: "8, 16, 32" }),
      ]),
    ).toBe(6);
  });

  it("returns 0 when nothing is set", () => {
    expect(gridTrialCount([row({})])).toBe(0);
  });
});
