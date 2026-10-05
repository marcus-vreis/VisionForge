import { describe, it, expect } from "vitest";
import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import { humanizeFieldPath } from "./useExperiment";

describe("humanizeFieldPath", () => {
  it("humanizes top-level fields", () => {
    expect(humanizeFieldPath(pt, ["body", "name"])).toBe("Nome");
    expect(humanizeFieldPath(pt, ["body", "task"])).toBe("Tipo de tarefa");
  });

  it("humanizes nested training fields", () => {
    expect(humanizeFieldPath(pt, ["body", "training", "learning_rate"])).toBe(
      "Treinamento › Learning Rate",
    );
    expect(humanizeFieldPath(pt, ["body", "training", "scheduler", "kind"])).toBe(
      "Treinamento › Scheduler › kind",
    );
  });

  it("renders preprocessing list indices as #N", () => {
    const path = ["body", "data", "preprocessing", "steps", 2, "radius"];
    expect(humanizeFieldPath(pt, path)).toBe(
      "Dataset › Pré-processamento › Filtro › #3 › radius",
    );
  });

  it("uses the first preprocessing index slot", () => {
    const path = ["body", "data", "preprocessing", "steps", 0, "kind"];
    expect(humanizeFieldPath(pt, path)).toBe(
      "Dataset › Pré-processamento › Filtro › #1 › kind",
    );
  });

  it("falls back to raw key for unknown fields", () => {
    expect(humanizeFieldPath(pt, ["body", "unknown_section", "foo"])).toBe(
      "unknown_section › foo",
    );
  });

  it("names a filter by its technical name, in either language", () => {
    const path = ["body", "data", "preprocessing", "steps", 0, "edges", "ksize"];
    expect(humanizeFieldPath(pt, path)).toBe(
      "Dataset › Pré-processamento › Filtro › #1 › Edges › ksize",
    );
    expect(humanizeFieldPath(en, path)).toBe(
      "Dataset › Preprocessing › Filter › #1 › Edges › ksize",
    );
  });

  it("writes the same path in English", () => {
    expect(humanizeFieldPath(en, ["body", "task"])).toBe("Task type");
    expect(humanizeFieldPath(en, ["body", "training", "learning_rate"])).toBe(
      "Training › Learning rate",
    );
    expect(humanizeFieldPath(en, ["body", "data", "base_dir"])).toBe(
      "Dataset › Base folder",
    );
    expect(humanizeFieldPath(en, ["body", "unknown_section", "foo"])).toBe(
      "unknown_section › foo",
    );
  });
});

describe("run messages", () => {
  it("counts the fields with a validation error, with the plural right in English", () => {
    expect(pt.experiment.validationFailed(3)).toBe(
      "3 campo(s) com erro de validação. Confira os destaques no formulário.",
    );
    expect(en.experiment.validationFailed(1)).toBe(
      "1 field has a validation error. Check the highlighted fields in the form.",
    );
    expect(en.experiment.validationFailed(3)).toBe(
      "3 fields have validation errors. Check the highlighted fields in the form.",
    );
  });
});
