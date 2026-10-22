import { afterEach, describe, expect, it, vi } from "vitest";

import {
  TASK_COLOR_PRESETS,
  isHexColor,
  readTaskColor,
  saveTaskColor,
  stageAccentVars,
  withTaskColors,
} from "./task-color";

function stubStorage(initial: Record<string, string> = {}) {
  const store = { ...initial };
  vi.stubGlobal("localStorage", {
    getItem: (k: string) => store[k] ?? null,
    setItem: (k: string, v: string) => {
      store[k] = v;
    },
    removeItem: (k: string) => {
      delete store[k];
    },
  });
  return store;
}

afterEach(() => vi.unstubAllGlobals());

describe("a researcher task's color (ADR-137)", () => {
  it("is remembered per task and forgotten on reset", () => {
    const store = stubStorage();
    expect(readTaskColor("counting")).toBeNull();
    saveTaskColor("counting", "#22c55e");
    expect(readTaskColor("counting")).toBe("#22c55e");
    expect(readTaskColor("other")).toBeNull();
    saveTaskColor("counting", null);
    expect(readTaskColor("counting")).toBeNull();
    expect(Object.keys(store)).toHaveLength(0);
  });

  it("ignores a stored value that is not a #rrggbb color", () => {
    stubStorage({ "vf.taskColor.counting": "red; background: url(x)" });
    expect(readTaskColor("counting")).toBeNull();
  });

  it("survives a blocked storage", () => {
    vi.stubGlobal("localStorage", {
      getItem: () => {
        throw new Error("blocked");
      },
      setItem: () => {
        throw new Error("blocked");
      },
      removeItem: () => {
        throw new Error("blocked");
      },
    });
    expect(readTaskColor("counting")).toBeNull();
    expect(() => saveTaskColor("counting", "#22c55e")).not.toThrow();
  });

  it("accepts only full hex colors, the format the registry declares", () => {
    expect(isHexColor("#2dd4bf")).toBe(true);
    expect(isHexColor("#2DD4BF")).toBe(true);
    expect(isHexColor("#fff")).toBe(false);
    expect(isHexColor("teal")).toBe(false);
    expect(TASK_COLOR_PRESETS.every(isHexColor)).toBe(true);
  });

  it("overrides only the researcher tasks that have a chosen color", () => {
    const tasks = [
      { key: "classification", accent: "#f16363", custom: false },
      { key: "counting", accent: "#2dd4bf", custom: true },
      { key: "vlm", accent: "#a855f7", custom: true },
    ];
    const merged = withTaskColors(tasks, { counting: "#22c55e", classification: "#000000" }, (t) => t.custom);
    expect(merged.map((t) => t.accent)).toEqual(["#f16363", "#22c55e", "#a855f7"]);
  });

  it("drives the stage's accent tokens from one color", () => {
    const vars = stageAccentVars("#22c55e") as Record<string, string>;
    expect(vars["--accent-vf"]).toBe("#22c55e");
    expect(vars["--accent-2"]).toContain("#22c55e");
    expect(vars["--accent-soft"]).toContain("#22c55e");
    expect(vars["--accent-glow"]).toContain("#22c55e");
  });
});
