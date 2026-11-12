import { describe, expect, it } from "vitest";
import { resultsOnTab } from "./results-tab";

const TABS = ["classification", "detection"];

describe("resultsOnTab", () => {
  it("shows the results on the tab of the task that ran", () => {
    expect(resultsOnTab("classification", "classification", TABS)).toBe(true);
  });

  it("hides them on another task's tab", () => {
    expect(resultsOnTab("classification", "detection", TABS)).toBe(false);
  });

  it("does not hide results whose task is unknown", () => {
    expect(resultsOnTab(null, "detection", TABS)).toBe(true);
  });

  it("does not hide results of a task that has no tab", () => {
    expect(resultsOnTab("removed-task", "detection", TABS)).toBe(true);
  });
});
