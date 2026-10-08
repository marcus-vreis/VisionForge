import { describe, expect, it } from "vitest";

import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import { stopLogLine } from "./stop-log";

describe("stopLogLine", () => {
  it("names the epoch a single run stopped at", () => {
    expect(stopLogLine(pt, { kind: "epoch", epoch: 3, total: 10 })).toMatch(/3\/10/);
    expect(stopLogLine(en, { kind: "epoch", epoch: 3, total: 10 })).toMatch(/3\/10/);
  });

  it("counts the units of a multi-unit job, not the epoch of the one in flight", () => {
    const summary = {
      kind: "units",
      unit: "fold",
      finished: 1,
      stopped: 1,
      planned: 3,
    } as const;
    expect(stopLogLine(pt, summary)).toMatch(/dobras concluídas: 1\/3/);
    expect(stopLogLine(en, summary)).toMatch(/folds finished: 1\/3/);
    expect(stopLogLine(pt, summary)).not.toMatch(/época|checkpoint/);
  });

  it("says what a stopped PatchCore kept", () => {
    expect(stopLogLine(en, { kind: "phase", bankKept: true })).toMatch(/memory bank saved/);
    expect(stopLogLine(en, { kind: "phase", bankKept: false })).toMatch(/nothing was kept/);
  });
});
