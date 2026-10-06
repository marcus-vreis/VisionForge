import { describe, expect, it } from "vitest";
import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import { phaseName } from "./training-phase";

// The labels core/anomaly_trainer.py sends, verbatim.
const SERVER_LABELS = ["extraindo features", "montando o banco", "pontuando"];

describe("phaseName", () => {
  it("leaves the three PatchCore labels as the server wrote them in Portuguese", () => {
    expect(SERVER_LABELS.map((label) => phaseName(pt, label))).toEqual(SERVER_LABELS);
  });

  it("words the three PatchCore labels in English", () => {
    expect(SERVER_LABELS.map((label) => phaseName(en, label))).toEqual([
      "extracting features",
      "building the memory bank",
      "scoring",
    ]);
  });

  it("shows a label it does not know exactly as received", () => {
    expect(phaseName(en, "calibrando")).toBe("calibrando");
    expect(phaseName(en, "constructor")).toBe("constructor");
  });
});
