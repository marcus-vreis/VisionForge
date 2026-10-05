import { describe, expect, it } from "vitest";

import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import { strategyLabel, taskLabel, waitedFor } from "./queue-format";

describe("taskLabel", () => {
  it("translates the built-in task keys", () => {
    expect(taskLabel(pt, "segmentation")).toBe("Segmentação");
    expect(taskLabel(en, "segmentation")).toBe("Segmentation");
  });

  it("shows a researcher's task under its own name", () => {
    expect(taskLabel(pt, "custom:example_counting")).toBe("example_counting");
    expect(taskLabel(en, "custom:example_counting")).toBe("example_counting");
  });

  it("falls back to the raw key rather than hiding an unknown task", () => {
    expect(taskLabel(pt, "something_new")).toBe("something_new");
  });
});

describe("strategyLabel", () => {
  it("translates the plain path submitted as a classification block", () => {
    expect(strategyLabel(pt, "classification")).toBe("treino simples");
    expect(strategyLabel(en, "classification")).toBe("single run");
    expect(strategyLabel(pt, "simple")).toBe(strategyLabel(pt, "classification"));
  });

  it("keeps the sweep mode visible", () => {
    expect(strategyLabel(pt, "sweep:optuna")).toBe("sweep · optuna");
    expect(strategyLabel(en, "sweep:optuna")).toBe("sweep · optuna");
  });

  it("translates the hyphenated strategy name", () => {
    expect(strategyLabel(pt, "replicated-comparison")).toBe("comparação replicada");
    expect(strategyLabel(en, "replicated-comparison")).toBe("replicated comparison");
  });

  it("falls back to the raw value", () => {
    expect(strategyLabel(pt, "brand_new")).toBe("brand_new");
  });

  it("names every strategy the backend sends, in both languages", () => {
    const expected: Record<string, [string, string]> = {
      simple: ["treino simples", "single run"],
      classification: ["treino simples", "single run"],
      cross_validation: ["K-fold", "K-fold"],
      cv: ["K-fold", "K-fold"],
      transfer_learning: ["transfer learning", "transfer learning"],
      grid_search: ["grid search", "grid search"],
      random_search: ["random search", "random search"],
      sweep: ["sweep", "sweep"],
      replicates: ["réplicas", "replicates"],
      comparison: ["comparação", "comparison"],
      "replicated-comparison": ["comparação replicada", "replicated comparison"],
    };
    for (const [strategy, [ptText, enText]] of Object.entries(expected)) {
      expect(strategyLabel(pt, strategy), strategy).toBe(ptText);
      expect(strategyLabel(en, strategy), strategy).toBe(enText);
    }
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
