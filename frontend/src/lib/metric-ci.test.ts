import { describe, expect, it } from "vitest";

import { formatWithCi, metricCi } from "./metric-ci";
import type { MetricCI } from "../types/run";

function ci(overrides: Partial<MetricCI> = {}): MetricCI {
  return {
    metric: "accuracy",
    value: 0.750625,
    ci_low: 0.729375,
    ci_high: 0.77128125,
    confidence: 0.95,
    n_resamples: 1000,
    n_samples: 1600,
    ...overrides,
  };
}

describe("metricCi", () => {
  it("strips the test_ prefix the metrics block uses", () => {
    expect(metricCi({ accuracy: ci() }, "test_accuracy")?.metric).toBe("accuracy");
  });

  it("also accepts the bare metric name", () => {
    expect(metricCi({ accuracy: ci() }, "accuracy")?.metric).toBe("accuracy");
  });

  it("returns undefined for a metric with no interval", () => {
    expect(metricCi({ accuracy: ci() }, "best_val_loss")).toBeUndefined();
  });

  it("returns undefined for a run written before ADR-074", () => {
    expect(metricCi(undefined, "test_accuracy")).toBeUndefined();
  });

  it("does not confuse a metric whose name merely contains test_", () => {
    expect(metricCi({ accuracy: ci() }, "latest_accuracy")).toBeUndefined();
  });
});

describe("formatWithCi", () => {
  it("appends the interval when present", () => {
    expect(formatWithCi(0.750625, ci())).toBe("0.7506 [0.7294, 0.7713]");
  });

  it("falls back to the bare value", () => {
    expect(formatWithCi(0.750625, undefined)).toBe("0.7506");
  });

  it("honours the digit count", () => {
    expect(formatWithCi(0.750625, ci(), 2)).toBe("0.75 [0.73, 0.77]");
  });
});
