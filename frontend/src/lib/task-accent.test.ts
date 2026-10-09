import { describe, expect, it } from "vitest";

import { accentForTask } from "./task-accent";

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
});
