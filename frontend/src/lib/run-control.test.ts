import { describe, expect, it } from "vitest";

import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import type { TrainingEvent } from "../types/run";
import {
  endedEarly,
  hasEnded,
  lastEpochOf,
  serverRunning,
  showQueueButton,
  stopMode,
  stopOutcome,
} from "./run-control";

const epoch = (n: number, total: number): TrainingEvent => ({
  event: "epoch_end",
  epoch: n,
  total_epochs: total,
  train_loss: 0.5,
  val_accuracy: 0.7,
});

const trialStart = (index: number, total: number): TrainingEvent => ({
  event: "trial_start",
  trial_index: index,
  total_trials: total,
  overrides: {},
  seed: 0,
});

describe("stopMode", () => {
  it("promises an epoch-boundary stop for the single runs the backend wires", () => {
    // What /api/queue reports: classification files its strategy under the block name.
    expect(stopMode({ task: "classification", strategy: "classification" })).toBe("epoch");
    expect(stopMode({ task: "classification", strategy: "transfer_learning" })).toBe("epoch");
    for (const task of ["detection", "regression", "segmentation", "anomaly"]) {
      expect(stopMode({ task, strategy: "simple" }), task).toBe("epoch");
    }
  });

  it("stops a classification search between trials", () => {
    expect(stopMode({ task: "classification", strategy: "grid_search" })).toBe("trial");
    expect(stopMode({ task: "classification", strategy: "random_search" })).toBe("trial");
  });

  it("promises nothing where the backend does not hand the token to the trainer", () => {
    const unwired: Array<[string, string]> = [
      ["classification", "cross_validation"],
      ["classification", "model_comparison"],
      // The standalone tasks' own sweeps, K-fold, comparisons and replicates.
      ["detection", "sweep:grid"],
      ["regression", "sweep:random"],
      ["segmentation", "cv"],
      ["segmentation", "comparison"],
      ["detection", "replicates"],
      ["regression", "replicated-comparison"],
      // A researcher's own task, whatever strategy it runs under.
      ["custom:counting", "simple"],
      ["custom:counting", "sweep:grid"],
      ["custom:counting", "replicates"],
    ];
    for (const [task, strategy] of unwired) {
      expect(stopMode({ task, strategy }), `${task} / ${strategy}`).toBe("none");
    }
  });

  it("does not stop a search the standalone tasks run as a sweep", () => {
    // Same words as the classification search, different path: only the task tells them apart.
    expect(stopMode({ task: "detection", strategy: "grid_search" })).toBe("none");
  });

  it("does not promise anything for a strategy it has never heard of", () => {
    expect(stopMode({ task: "classification", strategy: "brand_new" })).toBe("none");
  });

  it("assumes a plain run while the queue has not been read", () => {
    expect(stopMode(null)).toBe("epoch");
  });

  it("refuses a run that reports phases instead of epochs (PatchCore)", () => {
    expect(
      stopMode({ task: "anomaly", strategy: "simple" }, { phaseOnly: true }),
    ).toBe("none");
    expect(stopMode(null, { phaseOnly: true })).toBe("none");
  });
});

describe("showQueueButton", () => {
  it("is hidden on an idle server with nothing waiting", () => {
    expect(showQueueButton(0, false)).toBe(false);
  });

  it("shows while something waits", () => {
    expect(showQueueButton(2, false)).toBe(true);
  });

  it("shows while a job runs, even with nothing waiting", () => {
    expect(showQueueButton(0, true)).toBe(true);
  });
});

describe("serverRunning", () => {
  it("follows the tab's own run while it has one", () => {
    const base = { runActive: true, seededRunning: false };
    expect(serverRunning({ ...base, status: "running" })).toBe(true);
    // Waiting behind another job still means the server is busy.
    expect(serverRunning({ ...base, status: "queued" })).toBe(true);
    expect(serverRunning({ ...base, status: "completed" })).toBe(false);
    expect(serverRunning({ ...base, status: "failed" })).toBe(false);
  });

  it("does not let a stale snapshot override the tab's own finished run", () => {
    expect(
      serverRunning({ runActive: true, status: "completed", seededRunning: true }),
    ).toBe(false);
  });

  it("falls back to the queue snapshot when the tab has no run (after a reload)", () => {
    expect(
      serverRunning({ runActive: false, status: "idle", seededRunning: true }),
    ).toBe(true);
    expect(
      serverRunning({ runActive: false, status: "idle", seededRunning: false }),
    ).toBe(false);
  });
});

describe("endedEarly", () => {
  it("is true when a run's last epoch is short of what was configured", () => {
    expect(endedEarly([epoch(1, 4), epoch(2, 4), { event: "end", total_epochs: 2 }])).toBe(true);
  });

  it("is false when the run reached its last epoch", () => {
    expect(endedEarly([epoch(1, 2), epoch(2, 2), { event: "end", total_epochs: 2 }])).toBe(false);
  });

  it("is false for a stream with no epochs at all", () => {
    expect(endedEarly([])).toBe(false);
  });

  it("is true when a search skipped trials it had planned", () => {
    expect(
      endedEarly([
        trialStart(0, 4),
        epoch(2, 2),
        { event: "trial_end", total_epochs: 2, trial_index: 0, total_trials: 4 },
        { event: "end", total_epochs: 0, total_trials: 2 },
      ]),
    ).toBe(true);
  });

  it("is true when the search's last trial was cut mid-way", () => {
    expect(
      endedEarly([
        trialStart(0, 1),
        epoch(3, 10),
        { event: "end", total_epochs: 0, total_trials: 1 },
      ]),
    ).toBe(true);
  });

  it("is false when every trial ran to the end", () => {
    expect(
      endedEarly([
        trialStart(0, 2),
        epoch(2, 2),
        trialStart(1, 2),
        epoch(2, 2),
        { event: "end", total_epochs: 0, total_trials: 2 },
      ]),
    ).toBe(false);
  });
});

describe("lastEpochOf / hasEnded", () => {
  it("reads where the stream's last epoch stood", () => {
    expect(lastEpochOf([epoch(1, 6), epoch(3, 6)])).toEqual({ epoch: 3, total: 6 });
    expect(lastEpochOf([{ event: "start", total_epochs: 6 }])).toBeNull();
  });

  it("sees the terminal event", () => {
    expect(hasEnded([epoch(1, 2)])).toBe(false);
    expect(hasEnded([epoch(1, 2), { event: "end", total_epochs: 1 }])).toBe(true);
  });
});

describe("stopOutcome", () => {
  const stopped = [epoch(2, 4), { event: "end", total_epochs: 2 } as TrainingEvent];
  const complete = [epoch(4, 4), { event: "end", total_epochs: 4 } as TrainingEvent];

  it("reports nothing when no stop was asked", () => {
    expect(stopOutcome({ requested: false, ended: true, events: stopped })).toBe("none");
    expect(stopOutcome({ requested: false, ended: false, events: [] })).toBe("none");
  });

  it("is stopping while the run is still going", () => {
    expect(stopOutcome({ requested: true, ended: false, events: [epoch(2, 4)] })).toBe(
      "stopping",
    );
  });

  it("is stopped once the run ended short of its length", () => {
    expect(stopOutcome({ requested: true, ended: true, events: stopped })).toBe("stopped");
  });

  it("owns up when the stop arrived too late to change anything", () => {
    expect(stopOutcome({ requested: true, ended: true, events: complete })).toBe("too-late");
  });
});

describe("stop wording", () => {
  it("names the epoch the run stopped at, and survives not knowing it", () => {
    expect(pt.trainingOverlay.stoppedLog(3, 10)).toContain("3/10");
    expect(en.trainingOverlay.stoppedLog(3, 10)).toContain("3/10");
    expect(pt.trainingOverlay.stoppedLog(null, null)).not.toMatch(/null|\//);
    expect(en.trainingOverlay.stoppedLog(null, null)).not.toMatch(/null|\//);
  });
});
