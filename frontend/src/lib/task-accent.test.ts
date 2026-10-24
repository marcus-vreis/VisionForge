import { describe, expect, it } from "vitest";

import { TASK_ACCENT, TASK_ACCENT_LIGHT, accentForTask, customAccentMap, taskAccents } from "./task-accent";

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

describe("accentForTask with researcher tasks", () => {
  const custom = { "custom:foo": "#2dd4bf", "custom:bad": "teal" };

  it("uses the color the task registered or the researcher chose", () => {
    expect(accentForTask("custom:foo", "dark", custom)).toBe("#2dd4bf");
    expect(accentForTask("custom:foo", "light", custom)).toBe("#2dd4bf");
  });

  it("falls back to neutral for an unknown task or a color that is not #rrggbb", () => {
    expect(accentForTask("custom:other", "dark", custom)).toBe("var(--vf-text-muted)");
    expect(accentForTask("custom:bad", "dark", custom)).toBe("var(--vf-text-muted)");
  });

  it("leaves built-in tasks alone", () => {
    expect(accentForTask("detection", "dark", custom)).toBe("oklch(0.78 0.18 150)");
  });
});

describe("customAccentMap", () => {
  it("keys the colors by the task name a run records", () => {
    expect(
      customAccentMap([
        { key: "foo", accent: "#2dd4bf", custom: true },
        { key: "detection", accent: "#ff0000", custom: false },
      ]),
    ).toEqual({ "custom:foo": "#2dd4bf" });
  });
});
