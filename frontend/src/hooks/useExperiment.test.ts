import { describe, it, expect } from "vitest";
import { humanizeFieldPath } from "./useExperiment";

describe("humanizeFieldPath", () => {
  it("humanizes top-level fields", () => {
    expect(humanizeFieldPath(["body", "name"])).toBe("Nome");
    expect(humanizeFieldPath(["body", "task"])).toBe("Tipo de tarefa");
  });

  it("humanizes nested training fields", () => {
    expect(humanizeFieldPath(["body", "training", "learning_rate"])).toBe(
      "Treinamento › Learning Rate",
    );
    expect(humanizeFieldPath(["body", "training", "scheduler", "kind"])).toBe(
      "Treinamento › Scheduler › kind",
    );
  });

  it("renders preprocessing list indices as #N", () => {
    const path = ["body", "data", "preprocessing", "steps", 2, "radius"];
    expect(humanizeFieldPath(path)).toBe(
      "Dataset › Pré-processamento › Filtro › #3 › radius",
    );
  });

  it("uses the first preprocessing index slot", () => {
    const path = ["body", "data", "preprocessing", "steps", 0, "kind"];
    expect(humanizeFieldPath(path)).toBe(
      "Dataset › Pré-processamento › Filtro › #1 › kind",
    );
  });

  it("falls back to raw key for unknown fields", () => {
    expect(humanizeFieldPath(["body", "unknown_section", "foo"])).toBe(
      "unknown_section › foo",
    );
  });
});
