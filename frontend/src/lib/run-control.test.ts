import { describe, expect, it } from "vitest";

import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import type { TrainingEvent } from "../types/run";
import {
  epochLoop,
  hasEnded,
  reachedEpoch,
  runEnding,
  serverRunning,
  showQueueButton,
  stopMode,
  stopOutcome,
  type LocalRun,
} from "./run-control";

const start = (total: number): TrainingEvent => ({ event: "start", total_epochs: total });
const end = (total: number, trials?: number): TrainingEvent => ({
  event: "end",
  total_epochs: total,
  total_trials: trials,
});
const phase = (label = "buildingBank"): TrainingEvent => ({
  event: "phase",
  label,
  done: 1,
  total: 10,
});

/** What the overlay knows about its own run when the queue could not be read. */
const local = (blockKind: string, taskKey: string, loop: LocalRun["epochLoop"] = "yes"): LocalRun => ({
  blockKind,
  taskKey,
  epochLoop: loop,
});

/** An epoch report; inside a search the backend stamps it with its trial. */
const epoch = (
  n: number,
  total: number,
  trial?: { index: number; of: number },
): TrainingEvent => ({
  event: "epoch_end",
  epoch: n,
  total_epochs: total,
  train_loss: 0.5,
  val_accuracy: 0.7,
  ...(trial ? { trial_index: trial.index, total_trials: trial.of } : {}),
});

/** A search trial's end: its Trainer's `end`, rewritten, with how many epochs it ran. */
const trialEnd = (index: number, of: number, epochs: number): TrainingEvent => ({
  event: "trial_end",
  total_epochs: epochs,
  trial_index: index,
  total_trials: of,
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

  it("assumes a plain run when it knows nothing at all", () => {
    expect(stopMode(null)).toBe("epoch");
  });

  it("refuses a run that reports phases instead of epochs (PatchCore)", () => {
    const job = { task: "anomaly", strategy: "simple" };
    expect(stopMode(job, local("anomaly", "anomaly", "no"))).toBe("none");
    expect(stopMode(null, local("anomaly", "anomaly", "no"))).toBe("none");
  });

  describe("without the queue snapshot, from what the overlay itself knows", () => {
    it("does not offer a stop the server would ignore", () => {
      // The queue read failed: K-fold, comparison, replicates and custom tasks
      // must not fall back to a plain run.
      expect(stopMode(null, local("cross_validation", "classification"))).toBe("none");
      expect(stopMode(null, local("model_comparison", "classification"))).toBe("none");
      expect(stopMode(null, local("cross_validation", "regression"))).toBe("none");
      expect(stopMode(null, local("replicates", "detection"))).toBe("none");
      expect(stopMode(null, local("custom", "counting"))).toBe("none");
    });

    it("keeps the stop for the single runs and the classification searches", () => {
      expect(stopMode(null, local("classification", "classification"))).toBe("epoch");
      expect(stopMode(null, local("transfer_learning", "classification"))).toBe("epoch");
      for (const task of ["detection", "regression", "segmentation"]) {
        expect(stopMode(null, local(task, task)), task).toBe("epoch");
      }
      expect(stopMode(null, local("grid_search", "classification"))).toBe("trial");
      expect(stopMode(null, local("random_search", "classification"))).toBe("trial");
    });

    it("tells a classification search from another task's sweep by the tab it started in", () => {
      expect(stopMode(null, local("grid_search", "detection"))).toBe("none");
      expect(stopMode(null, local("random_search", "counting"))).toBe("none");
    });

    it("is overruled by the snapshot when there is one", () => {
      expect(
        stopMode(
          { task: "classification", strategy: "cross_validation" },
          local("classification", "classification"),
        ),
      ).toBe("none");
      expect(
        stopMode({ task: "detection", strategy: "simple" }, local("custom", "counting")),
      ).toBe("epoch");
    });

    it("does not promise anything for a block it has never heard of", () => {
      expect(stopMode(null, local("brand_new", "classification"))).toBe("none");
    });
  });

  describe("anomaly runs, whose model decides whether there are epochs at all", () => {
    it("waits for proof of an epoch loop before offering the stop", () => {
      const job = { task: "anomaly", strategy: "simple" };
      expect(stopMode(job, local("anomaly", "anomaly", "unknown"))).toBe("unconfirmed");
      expect(stopMode(null, local("anomaly", "anomaly", "unknown"))).toBe("unconfirmed");
      expect(stopMode(job, local("anomaly", "anomaly", "yes"))).toBe("epoch");
    });

    it("does not make the other tasks wait", () => {
      expect(stopMode(null, local("detection", "detection", "unknown"))).toBe("epoch");
      expect(
        stopMode({ task: "classification", strategy: "classification" }, local("classification", "classification", "unknown")),
      ).toBe("epoch");
    });
  });
});

describe("epochLoop", () => {
  it("is unknown until the run says how it works", () => {
    expect(epochLoop([])).toBe("unknown");
    // One configured epoch cannot tell an autoencoder from PatchCore's single step.
    expect(epochLoop([start(1)])).toBe("unknown");
  });

  it("sees an epoch loop in several configured epochs or in an epoch report", () => {
    expect(epochLoop([start(30)])).toBe("yes");
    expect(epochLoop([start(1), epoch(1, 1)])).toBe("yes");
  });

  it("sees no epoch loop once phases are reported, even after the closing epoch", () => {
    expect(epochLoop([start(1), phase()])).toBe("no");
    // PatchCore closes with one epoch_end after its phases.
    expect(epochLoop([start(1), phase(), epoch(1, 1)])).toBe("no");
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

describe("runEnding", () => {
  it("is early when a run's last epoch is short of what was configured", () => {
    expect(runEnding([epoch(1, 4), epoch(2, 4), end(2)])).toBe("early");
  });

  it("is complete only when the last epoch reported reached its total", () => {
    expect(runEnding([epoch(1, 2), epoch(2, 2), end(2)])).toBe("complete");
  });

  it("does not know anything from an empty stream", () => {
    expect(runEnding([])).toBe("unknown");
  });

  it("reads a resumed run that stopped before its first new epoch", () => {
    // The history of the earlier pass is not replayed as epoch_end: all the
    // stream carries is how long the run was meant to be and how far it got.
    expect(runEnding([start(20), end(5)])).toBe("early");
  });

  it("reads a fresh run that stopped before its first epoch", () => {
    expect(runEnding([start(8), end(0)])).toBe("early");
  });

  it("does not call a stream with no epoch report complete, whatever its start and end say", () => {
    expect(runEnding([start(5), end(5)])).toBe("unknown");
    expect(runEnding([end(5)])).toBe("unknown");
  });

  it("is early when a search skipped trials it had planned", () => {
    expect(
      runEnding([
        trialStart(0, 4),
        epoch(2, 2, { index: 0, of: 4 }),
        trialEnd(0, 4, 2),
        end(0, 2),
      ]),
    ).toBe("early");
  });

  it("is early when the search's last trial was cut mid-way", () => {
    expect(
      runEnding([trialStart(0, 1), epoch(3, 10, { index: 0, of: 1 }), end(0, 1)]),
    ).toBe("early");
  });

  it("is complete when every trial ran to the end", () => {
    expect(
      runEnding([
        trialStart(0, 2),
        epoch(2, 2, { index: 0, of: 2 }),
        trialEnd(0, 2, 2),
        trialStart(1, 2),
        epoch(2, 2, { index: 1, of: 2 }),
        trialEnd(1, 2, 2),
        end(0, 2),
      ]),
    ).toBe("complete");
  });

  it("is early when the last trial was stopped before it trained an epoch", () => {
    // The search still ends with every trial counted, and the last epoch the
    // stream carries is the previous trial's final one: reading that as the
    // run's last epoch would call a stopped search finished.
    expect(
      runEnding([
        trialStart(0, 2),
        epoch(2, 2, { index: 0, of: 2 }),
        trialEnd(0, 2, 2),
        trialStart(1, 2),
        trialEnd(1, 2, 0),
        end(0, 2),
      ]),
    ).toBe("early");
  });

  it("is early when a middle trial stopped before it trained an epoch", () => {
    expect(
      runEnding([
        trialStart(0, 3),
        epoch(2, 2, { index: 0, of: 3 }),
        trialEnd(0, 3, 2),
        trialStart(1, 3),
        trialEnd(1, 3, 0),
        end(0, 2),
      ]),
    ).toBe("early");
  });

  it("does not call a search complete when its last trial reported no epoch", () => {
    // Nothing says the last trial trained (and it did not stop at an epoch of
    // its own), so nothing is claimed.
    expect(
      runEnding([
        trialStart(0, 2),
        epoch(2, 2, { index: 0, of: 2 }),
        trialEnd(0, 2, 2),
        trialStart(1, 2),
        end(0, 2),
      ]),
    ).toBe("unknown");
  });

  it("tells the trial of an epoch from the last trial_start when it is not stamped", () => {
    expect(
      runEnding([
        trialStart(0, 2),
        epoch(2, 2),
        trialStart(1, 2),
        epoch(1, 2),
        end(0, 2),
      ]),
    ).toBe("early");
    expect(
      runEnding([
        trialStart(0, 2),
        epoch(2, 2),
        trialStart(1, 2),
        epoch(2, 2),
        end(0, 2),
      ]),
    ).toBe("complete");
  });

  it("does not read a search's end as a single run's", () => {
    // A search's end event reports trials, and its `total_epochs` is 0.
    expect(runEnding([start(4), trialStart(0, 1), end(0, 1)])).toBe("unknown");
  });
});

describe("reachedEpoch / hasEnded", () => {
  it("reads where the stream's last epoch stood", () => {
    expect(reachedEpoch([epoch(1, 6), epoch(3, 6)])).toEqual({ epoch: 3, total: 6 });
  });

  it("falls back to the run's own start and end when no epoch was reported", () => {
    expect(reachedEpoch([start(20), end(5)])).toEqual({ epoch: 5, total: 20 });
    expect(reachedEpoch([start(8), end(0)])).toEqual({ epoch: 0, total: 8 });
  });

  it("does not guess from a stream that says nothing about epochs", () => {
    expect(reachedEpoch([start(6)])).toBeNull();
    expect(reachedEpoch([])).toBeNull();
    expect(reachedEpoch([start(4), trialStart(0, 1), end(0, 1)])).toBeNull();
  });

  it("does not report the previous trial's last epoch as where a stopped search stood", () => {
    expect(
      reachedEpoch([
        trialStart(0, 2),
        epoch(2, 2, { index: 0, of: 2 }),
        trialEnd(0, 2, 2),
        trialStart(1, 2),
        epoch(3, 10, { index: 1, of: 2 }),
        end(0, 2),
      ]),
    ).toEqual({ epoch: 3, total: 10 });
    expect(
      reachedEpoch([
        trialStart(0, 2),
        epoch(2, 2, { index: 0, of: 2 }),
        trialEnd(0, 2, 2),
        trialStart(1, 2),
        trialEnd(1, 2, 0),
        end(0, 2),
      ]),
    ).toEqual({ epoch: 0, total: 2 });
  });

  it("sees the terminal event", () => {
    expect(hasEnded([epoch(1, 2)])).toBe(false);
    expect(hasEnded([epoch(1, 2), end(1)])).toBe(true);
  });
});

describe("stopOutcome", () => {
  const stopped = [epoch(2, 4), end(2)];
  const complete = [epoch(4, 4), end(4)];

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

  it("is stopped for a resumed run stopped before it trained another epoch", () => {
    expect(
      stopOutcome({ requested: true, ended: true, events: [start(20), end(5)] }),
    ).toBe("stopped");
  });

  it("is stopped for a fresh run stopped before its first epoch", () => {
    expect(
      stopOutcome({ requested: true, ended: true, events: [start(8), end(0)] }),
    ).toBe("stopped");
  });

  it("owns up when the stop arrived too late to change anything", () => {
    expect(stopOutcome({ requested: true, ended: true, events: complete })).toBe("too-late");
  });

  it("never says the run finished normally without an epoch that reached its total", () => {
    expect(stopOutcome({ requested: true, ended: true, events: [] })).toBe("unknown");
    expect(
      stopOutcome({ requested: true, ended: true, events: [start(5), end(5)] }),
    ).toBe("unknown");
  });
});

describe("stop wording", () => {
  it("names the epoch the run stopped at, and survives not knowing it", () => {
    expect(pt.trainingOverlay.stoppedLog(3, 10)).toContain("3/10");
    expect(en.trainingOverlay.stoppedLog(3, 10)).toContain("3/10");
    expect(pt.trainingOverlay.stoppedLog(null, null)).not.toMatch(/null|\//);
    expect(en.trainingOverlay.stoppedLog(null, null)).not.toMatch(/null|\//);
  });

  it("does not claim a checkpoint for a run that stopped before its first epoch", () => {
    expect(pt.trainingOverlay.stoppedLog(0, 8)).not.toMatch(/checkpoint|0\/8/);
    expect(en.trainingOverlay.stoppedLog(0, 8)).not.toMatch(/checkpoint|0\/8/);
  });

  it("speaks of the run, not of its kind, in the right gender", () => {
    // "Esta execução … interrompida", not "Este tipo de execução … interrompida".
    expect(pt.trainingOverlay.stopUnavailable).toMatch(/^Esta execução/);
    expect(pt.queueOverlay.stopUnavailable).toMatch(/^Esta execução/);
  });
});
