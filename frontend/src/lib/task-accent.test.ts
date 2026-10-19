import { describe, expect, it } from "vitest";

import { TASK_ACCENT, TASK_ACCENT_LIGHT, accentForTask, taskAccents } from "./task-accent";

describe("accentForTask", () => {
  it("colors each task as the interface does", () => {
    expect(accentForTask("classification")).toBe("oklch(0.74 0.18 22)");
    expect(accentForTask("detection")).toBe("oklch(0.78 0.18 150)");
    expect(accentForTask("regression")).toBe("oklch(0.74 0.16 240)");
    expect(accentForTask("segmentation")).toBe("oklch(0.74 0.18 305)");
  });

  it("reads a classification problem type as classification", () => {
    expect(accentForTask("binary")).toBe("oklch(0.74 0.18 22)");
    expect(accentForTask("multiclass")).toBe("oklch(0.74 0.18 22)");
  });

  it("gives a researcher's own task no color of its own", () => {
    expect(accentForTask("custom:foo")).toBe("var(--vf-text-muted)");
    expect(accentForTask(undefined)).toBe("var(--vf-text-muted)");
  });

  it("keeps the dark colors when no theme or the dark one is given", () => {
    for (const task of ["classification", "binary", "detection", "regression", "segmentation", "anomaly"]) {
      expect(accentForTask(task, "dark")).toBe(accentForTask(task));
    }
    expect(taskAccents()).toBe(TASK_ACCENT);
    expect(taskAccents("dark")).toBe(TASK_ACCENT);
  });

  it("darkens each task in the light theme", () => {
    expect(accentForTask("classification", "light")).toBe("oklch(0.56 0.18 22)");
    expect(accentForTask("multiclass", "light")).toBe("oklch(0.56 0.18 22)");
    expect(accentForTask("detection", "light")).toBe("oklch(0.52 0.15 150)");
    expect(accentForTask("regression", "light")).toBe("oklch(0.52 0.16 240)");
    expect(accentForTask("segmentation", "light")).toBe("oklch(0.52 0.18 305)");
    expect(accentForTask("anomaly", "light")).toBe("oklch(0.50 0.15 75)");
    expect(taskAccents("light")).toBe(TASK_ACCENT_LIGHT);
  });

  it("gives a researcher's own task the neutral color in the light theme too", () => {
    expect(accentForTask("custom:foo", "light")).toBe("var(--vf-text-muted)");
    expect(accentForTask(null, "light")).toBe("var(--vf-text-muted)");
  });

  it("colors every task in both themes with a plain oklch literal", () => {
    expect(Object.keys(TASK_ACCENT_LIGHT).sort()).toEqual(Object.keys(TASK_ACCENT).sort());
    for (const color of [...Object.values(TASK_ACCENT), ...Object.values(TASK_ACCENT_LIGHT)]) {
      expect(color).toMatch(/^oklch\(0\.\d+ 0\.\d+ \d+\)$/);
    }
  });
});
