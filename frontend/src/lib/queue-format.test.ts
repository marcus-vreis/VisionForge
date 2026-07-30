import { describe, expect, it } from "vitest";

import { strategyLabel, taskLabel, waitedFor } from "./queue-format";

describe("taskLabel", () => {
  it("translates the built-in task keys", () => {
    expect(taskLabel("segmentation")).toBe("Segmentação");
  });

  it("shows a researcher's task under its own name", () => {
    expect(taskLabel("custom:example_counting")).toBe("example_counting");
  });

  it("falls back to the raw key rather than hiding an unknown task", () => {
    expect(taskLabel("something_new")).toBe("something_new");
  });
});

describe("strategyLabel", () => {
  it("translates the plain path submitted as a classification block", () => {
    expect(strategyLabel("classification")).toBe("treino simples");
  });

  it("keeps the sweep mode visible", () => {
    expect(strategyLabel("sweep:optuna")).toBe("sweep · optuna");
  });

  it("translates the hyphenated strategy name", () => {
    expect(strategyLabel("replicated-comparison")).toBe("comparação replicada");
  });

  it("falls back to the raw value", () => {
    expect(strategyLabel("brand_new")).toBe("brand_new");
  });
});

describe("waitedFor", () => {
  const submitted = "2026-07-29T12:00:00.000Z";
  const at = (offsetSeconds: number) =>
    Date.parse(submitted) + offsetSeconds * 1000;

  it("uses seconds under a minute", () => {
    expect(waitedFor(submitted, at(42))).toBe("42s");
  });

  it("uses minutes under an hour", () => {
    expect(waitedFor(submitted, at(20 * 60))).toBe("20min");
  });

  it("uses hours beyond that", () => {
    expect(waitedFor(submitted, at(5400))).toBe("1.5h");
  });

  it("never reports negative time when the clocks disagree", () => {
    expect(waitedFor(submitted, at(-30))).toBe("0s");
  });
});
