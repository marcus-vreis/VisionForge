import { describe, expect, it } from "vitest";

import { en } from "../../i18n/en";
import { pt } from "../../i18n/pt";
import { advanceLatch, gateOpen, guideFacts, openLatch } from "./gates";
import { firstTrainingGuide } from "./first-training";
import { interfaceTour } from "./interface-tour";
import type { GuideFacts, GuideStep } from "./types";

const start = { event: "start" };
const epoch = { event: "epoch_end" };
const end = { event: "end" };

const facts = (over: Partial<GuideFacts> = {}): GuideFacts => ({
  datasetPath: "",
  trainingStarted: false,
  trainingEnded: false,
  ...over,
});

describe("the event latch", () => {
  it("starts closed on an empty stream", () => {
    const latch = openLatch([]);

    expect(guideFacts(latch, "")).toEqual(facts());
  });

  it("opens the start gate when a start appears, and the end gate at its end", () => {
    let latch = openLatch([]);

    latch = advanceLatch(latch, [start]);
    expect(guideFacts(latch, "").trainingStarted).toBe(true);
    expect(guideFacts(latch, "").trainingEnded).toBe(false);

    latch = advanceLatch(latch, [start, epoch, end]);
    expect(guideFacts(latch, "").trainingEnded).toBe(true);
  });

  it("does not count a finished run that was already on screen", () => {
    // The hook keeps the last run's events until the next submission.
    const latch = openLatch([start, epoch, end]);

    expect(guideFacts(latch, "")).toEqual(facts());
    expect(guideFacts(advanceLatch(latch, [start, epoch, end]), "")).toEqual(facts());
  });

  it("counts the next run after the stale one is cleared by a submission", () => {
    let latch = openLatch([start, end]);

    latch = advanceLatch(latch, []); // submit() empties the list
    expect(guideFacts(latch, "").trainingStarted).toBe(false);
    latch = advanceLatch(latch, [start]);
    expect(guideFacts(latch, "").trainingStarted).toBe(true);
    expect(guideFacts(latch, "").trainingEnded).toBe(false);
    latch = advanceLatch(latch, [start, end]);
    expect(guideFacts(latch, "").trainingEnded).toBe(true);
  });

  it("does not count the end of a run that was already in flight", () => {
    let latch = openLatch([start, epoch]);

    latch = advanceLatch(latch, [start, epoch, end]);

    expect(guideFacts(latch, "")).toEqual(facts());
  });

  it("counts a start and an end that arrive in the same look", () => {
    const latch = advanceLatch(openLatch([]), [start, end]);

    expect(guideFacts(latch, "")).toMatchObject({
      trainingStarted: true,
      trainingEnded: true,
    });
  });

  it("keeps a gate open once it opened, even if the stream is cleared after", () => {
    let latch = advanceLatch(openLatch([]), [start, end]);

    latch = advanceLatch(latch, []);

    expect(guideFacts(latch, "").trainingEnded).toBe(true);
  });

  it("returns the same object when nothing changed", () => {
    const latch = openLatch([start]);

    expect(advanceLatch(latch, [start])).toBe(latch);
  });

  it("carries the dataset path through unchanged", () => {
    expect(guideFacts(openLatch([]), "C:/data").datasetPath).toBe("C:/data");
  });
});

describe("gateOpen", () => {
  const plain: GuideStep = { title: "t", body: "b" };

  it("is open for a step with no gate", () => {
    expect(gateOpen(plain, facts())).toBe(true);
  });

  it("follows the step's predicate", () => {
    const step: GuideStep = { ...plain, waitFor: (f) => f.trainingStarted };

    expect(gateOpen(step, facts())).toBe(false);
    expect(gateOpen(step, facts({ trainingStarted: true }))).toBe(true);
  });
});

describe("the tour keeps playing as before", () => {
  it("has no gate, action, entry hook or floating card on any step", () => {
    for (const dict of [pt, en]) {
      for (const step of interfaceTour.steps(dict)) {
        expect(step.waitFor).toBeUndefined();
        expect(step.waitHint).toBeUndefined();
        expect(step.action).toBeUndefined();
        expect(step.onEnter).toBeUndefined();
        expect(step.floating).toBeUndefined();
      }
    }
  });
});

describe("the gates of Primeiro treino", () => {
  const steps = firstTrainingGuide.steps(pt);
  const gated = steps.filter((s) => s.waitFor);

  it("has exactly three: dataset set, training started, training ended", () => {
    expect(gated).toHaveLength(3);
    const [dataset, started, ended] = gated.map((s) => s.waitFor!);

    expect(dataset(facts())).toBe(false);
    expect(dataset(facts({ datasetPath: "   " }))).toBe(false);
    expect(dataset(facts({ datasetPath: "C:/datasets/exemplo" }))).toBe(true);

    expect(started(facts())).toBe(false);
    expect(started(facts({ trainingStarted: true }))).toBe(true);
    expect(started(facts({ trainingEnded: true }))).toBe(false);

    expect(ended(facts({ trainingStarted: true }))).toBe(false);
    expect(ended(facts({ trainingStarted: true, trainingEnded: true }))).toBe(true);
  });

  it("says what each gate is waiting for", () => {
    for (const dict of [pt, en]) {
      for (const step of firstTrainingGuide.steps(dict)) {
        if (step.waitFor) expect(step.waitHint?.length ?? 0).toBeGreaterThan(10);
      }
    }
  });

  it("opens in order: nothing is open before the dataset is set", () => {
    // The path alone opens the first gate and none of the later ones.
    const only = facts({ datasetPath: "C:/x" });

    expect(gated.map((s) => gateOpen(s, only))).toEqual([true, false, false]);
  });
});
