import { describe, it, expect } from "vitest";
import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import { humanizeFieldPath } from "./useExperiment";

describe("humanizeFieldPath", () => {
  it("humanizes top-level fields", () => {
    expect(humanizeFieldPath(pt, ["body", "name"])).toBe("Nome do experimento");
    expect(humanizeFieldPath(pt, ["body", "task"])).toBe("Tipo de tarefa");
  });

  it("humanizes nested training fields", () => {
    expect(humanizeFieldPath(pt, ["body", "training", "learning_rate"])).toBe(
      "Treinamento › Learning rate",
    );
    expect(humanizeFieldPath(pt, ["body", "training", "scheduler", "kind"])).toBe(
      "Treinamento › Scheduler › Tipo",
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
      "Dataset › Pré-processamento › Filtro › #1 › Tipo",
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
      "Dataset › Pré-processamento › Filtro › #1 › Edges (Sobel) › ksize",
    );
    expect(humanizeFieldPath(en, path)).toBe(
      "Dataset › Preprocessing › Filter › #1 › Edges (Sobel) › ksize",
    );
  });

  it("names a field as the form does, in both languages", () => {
    // The errors and the form share one set of names (paramPanel.*Labels), so a
    // message never calls a field something the field is not labelled.
    expect(humanizeFieldPath(pt, ["body", "data", "base_dir"])).toBe("Dataset › Pasta base");
    expect(humanizeFieldPath(en, ["body", "name"])).toBe("Experiment name");
    expect(humanizeFieldPath(en, ["body", "training", "early_stopping_patience"])).toBe(
      "Training › Early stop (patience)",
    );
    expect(humanizeFieldPath(en, ["body", "data", "transforms", "rotation_degrees"])).toBe(
      "Dataset › Transforms › Rotation (degrees)",
    );
    expect(humanizeFieldPath(en, ["body", "model", "weights_path"])).toBe(
      `Model › ${en.paramPanel.weights.label}`,
    );
    expect(humanizeFieldPath(pt, ["body", "model", "weights_path"])).toBe(
      `Modelo › ${pt.paramPanel.weights.label}`,
    );
  });

  it("prefers the path-qualified name: model.name is the architecture", () => {
    expect(humanizeFieldPath(en, ["body", "model", "name"])).toBe("Model › Architecture");
    expect(humanizeFieldPath(pt, ["body", "model", "name"])).toBe("Modelo › Arquitetura");
    expect(humanizeFieldPath(en, ["body", "name"])).toBe("Experiment name");
  });

  it("keeps in experiment.sections only the segments the form has no label for", () => {
    for (const dict of [pt, en]) {
      const own = Object.keys(dict.experiment.sections);
      expect(own.sort()).toEqual(["device", "preprocessing", "scheduler", "steps"]);
      for (const key of own) {
        expect(key in dict.paramPanel.sectionLabels, key).toBe(false);
        expect(key in dict.paramPanel.fieldLabels, key).toBe(false);
      }
    }
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
  it("counts the fields with a validation error, with the plural right in both languages", () => {
    expect(pt.experiment.validationFailed(1)).toBe(
      "1 campo com erro de validação. Confira o destaque no formulário.",
    );
    expect(pt.experiment.validationFailed(3)).toBe(
      "3 campos com erro de validação. Confira os destaques no formulário.",
    );
    expect(en.experiment.validationFailed(1)).toBe(
      "1 field has a validation error. Check the highlighted field in the form.",
    );
    expect(en.experiment.validationFailed(3)).toBe(
      "3 fields have validation errors. Check the highlighted fields in the form.",
    );
  });
});
