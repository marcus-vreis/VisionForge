import { describe, expect, it } from "vitest";

import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import { taskDefinitions } from "./tasks";

describe("taskDefinitions", () => {
  it("lists the five built-in tasks in the same order in both languages", () => {
    const keys = ["classification", "detection", "regression", "segmentation", "anomaly"];
    expect(taskDefinitions(pt).map((task) => task.key)).toEqual(keys);
    expect(taskDefinitions(en).map((task) => task.key)).toEqual(keys);
  });

  it("names each tab and its hero sentence in the active language", () => {
    const byKey = (dict: typeof pt) =>
      Object.fromEntries(
        taskDefinitions(dict).map((task) => [task.key, [task.label, task.description]]),
      );
    expect(byKey(pt).detection).toEqual([
      "Detecção de Objeto",
      "Localize e classifique objetos com bounding boxes",
    ]);
    expect(byKey(en).detection).toEqual([
      "Object detection",
      "Locate and classify objects with bounding boxes",
    ]);
    expect(byKey(pt).anomaly[0]).toBe("Anomalia");
    expect(byKey(en).anomaly[0]).toBe("Anomaly");
  });

  it("leaves no text empty or undefined, in either language", () => {
    for (const dict of [pt, en]) {
      for (const task of taskDefinitions(dict)) {
        for (const text of [task.label, task.description]) {
          expect(text, task.key).toBeTruthy();
          expect(text.trim(), task.key).not.toBe("");
          expect(text, task.key).not.toContain("undefined");
        }
      }
    }
  });

  it("keeps accents and keys the same in every language", () => {
    const shape = (dict: typeof pt) =>
      taskDefinitions(dict).map((task) => ({
        key: task.key,
        short: task.short,
        accent: task.accent,
      }));
    expect(shape(en)).toEqual(shape(pt));
  });
});
