import { describe, expect, it } from "vitest";

import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import { plannedUnits, unitPlan } from "./unit-plan";

describe("unitPlan", () => {
  it("counts against the planned units when more were planned than ran", () => {
    // A K-fold of 5 stopped in its first fold: 1 ran, 4 never started.
    expect(unitPlan(1, 5)).toEqual({ total: 5, notRun: 4 });
  });

  it("is the units that ran when all of them did", () => {
    expect(unitPlan(5, 5)).toEqual({ total: 5, notRun: 0 });
  });

  it("falls back to the units that ran when the plan is not known", () => {
    expect(unitPlan(3, null)).toEqual({ total: 3, notRun: 0 });
    expect(unitPlan(3, undefined)).toEqual({ total: 3, notRun: 0 });
  });

  it("does not trust a plan smaller than what ran", () => {
    // A stale or wrong figure must not make a count read as "3/2".
    expect(unitPlan(3, 2)).toEqual({ total: 3, notRun: 0 });
    expect(unitPlan(3, Number.NaN)).toEqual({ total: 3, notRun: 0 });
    expect(unitPlan(3, 0)).toEqual({ total: 3, notRun: 0 });
  });
});

describe("plannedUnits", () => {
  it("reads the K-fold report's own fold count", () => {
    expect(plannedUnits({ n_folds: 5, fold_results: [] }, 3)).toBe(5);
  });

  it("reads a standalone sweep's planned trials", () => {
    expect(plannedUnits({ planned_trials: 6, trials: [] }, 4)).toBe(6);
  });

  it("reads the seeds a replicate set was asked for", () => {
    expect(plannedUnits({ seeds: [0, 1, 2, 3], trials: [{}] }, null)).toBe(4);
  });

  it("falls back to what was submitted, for a report that does not say", () => {
    // The classification K-fold, comparison and grid search list only what ran.
    expect(plannedUnits({ fold_results: [] }, 5)).toBe(5);
    expect(plannedUnits({ best_trial: null, total_trials: 1 }, 12)).toBe(12);
  });

  it("is unknown when nothing says", () => {
    expect(plannedUnits({ fold_results: [] }, null)).toBeNull();
    expect(plannedUnits({}, undefined)).toBeNull();
  });
});

describe("the not-run note", () => {
  it("counts the units that never started, in both languages", () => {
    expect(pt.resultsView.notRun(4)).toBe(" · 4 não rodaram");
    expect(pt.resultsView.notRun(1)).toBe(" · 1 não rodou");
    expect(en.resultsView.notRun(4)).toBe(" · 4 not run");
    expect(en.resultsView.notRun(1)).toBe(" · 1 not run");
  });
});
