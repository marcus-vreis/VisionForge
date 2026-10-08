import { describe, expect, it } from "vitest";

import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import type { StopPoint, TrainingEvent } from "../types/run";
import {
  awaitingReport,
  epochLoop,
  hasEnded,
  isCustomJob,
  reachedEpoch,
  runEnding,
  serverRunning,
  showQueueButton,
  stopMode,
  stopOutcome,
  stopSummary,
  stoppedFraction,
  unitCounts,
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
const local = (blockKind: string, loop: LocalRun["epochLoop"] = "yes"): LocalRun => ({
  blockKind,
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

const POINTS: StopPoint[] = ["epoch", "trial", "fold", "model", "replicate", "phase"];

describe("stopMode", () => {
  describe("with the queue snapshot", () => {
    it("takes where the job stops from the snapshot itself", () => {
      for (const stop_at of POINTS) {
        expect(stopMode({ task: "classification", strategy: "x", stop_at }), stop_at).toBe(
          stop_at,
        );
      }
    });

    it("reads a null stop_at as a running job that cannot be stopped", () => {
      expect(stopMode({ task: "custom:counting", strategy: "simple", stop_at: null })).toBe(
        "none",
      );
    });

    it("trusts the snapshot over what the sheet recorded for itself", () => {
      expect(
        stopMode(
          { task: "classification", strategy: "cross_validation", stop_at: "fold" },
          local("classification"),
        ),
      ).toBe("fold");
      // The snapshot says PatchCore: no need to wait for the stream to show it.
      expect(
        stopMode(
          { task: "anomaly", strategy: "simple", stop_at: "phase" },
          local("anomaly", "unknown"),
        ),
      ).toBe("phase");
    });
  });

  describe("from the strategy, when the snapshot does not say where", () => {
    // The server's own table (routes._STOP_POINTS), mirrored.
    it("mirrors the server's mapping", () => {
      const cases: Array<[string, StopPoint]> = [
        ["simple", "epoch"],
        ["classification", "epoch"],
        ["transfer_learning", "epoch"],
        ["batch_prediction", "epoch"],
        ["export_onnx", "epoch"],
        ["grid_search", "trial"],
        ["random_search", "trial"],
        ["sweep:grid", "trial"],
        ["sweep:optuna", "trial"],
        ["cross_validation", "fold"],
        ["cv", "fold"],
        ["model_comparison", "model"],
        ["comparison", "model"],
        ["replicates", "replicate"],
        ["replicated-comparison", "replicate"],
      ];
      for (const [strategy, point] of cases) {
        expect(stopMode({ task: "classification", strategy }), strategy).toBe(point);
      }
    });

    it("does not claim a stop for a strategy it has never heard of", () => {
      expect(stopMode({ task: "classification", strategy: "brand_new" })).toBe("none");
    });

    it("cannot tell which level a custom task is, so it waits for the snapshot", () => {
      expect(stopMode({ task: "custom:counting", strategy: "simple" })).toBe("unconfirmed");
      // Its sweeps and replicates stop between units whichever level it is.
      expect(stopMode({ task: "custom:counting", strategy: "sweep:grid" })).toBe("trial");
      expect(stopMode({ task: "custom:counting", strategy: "replicates" })).toBe("replicate");
    });
  });

  describe("without the snapshot, from what the overlay itself knows", () => {
    it("stops every kind of run the server now stops", () => {
      const cases: Array<[string, StopPoint]> = [
        ["classification", "epoch"],
        ["transfer_learning", "epoch"],
        ["detection", "epoch"],
        ["regression", "epoch"],
        ["segmentation", "epoch"],
        ["grid_search", "trial"],
        ["random_search", "trial"],
        ["cross_validation", "fold"],
        ["model_comparison", "model"],
        ["replicates", "replicate"],
      ];
      for (const [blockKind, point] of cases) {
        expect(stopMode(null, local(blockKind)), blockKind).toBe(point);
      }
    });

    it("waits for the snapshot on a custom task, whose level it cannot know", () => {
      expect(stopMode(null, local("custom"))).toBe("unconfirmed");
    });

    it("does not claim a stop for a block it has never heard of", () => {
      expect(stopMode(null, local("brand_new"))).toBe("none");
    });

    it("assumes a plain run when it knows nothing at all", () => {
      expect(stopMode(null)).toBe("epoch");
    });

    it("tells an anomaly run's kind from its stream", () => {
      const job = { task: "anomaly", strategy: "simple" };
      expect(stopMode(job, local("anomaly", "yes"))).toBe("epoch");
      expect(stopMode(job, local("anomaly", "no"))).toBe("phase");
      expect(stopMode(null, local("anomaly", "no"))).toBe("phase");
      // Not offered until the stream says which.
      expect(stopMode(job, local("anomaly", "unknown"))).toBe("unconfirmed");
      expect(stopMode(null, local("anomaly", "unknown"))).toBe("unconfirmed");
    });

    it("does not make the other tasks wait for their stream", () => {
      expect(stopMode(null, local("detection", "unknown"))).toBe("epoch");
      expect(
        stopMode({ task: "classification", strategy: "classification" }, local("classification", "unknown")),
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

  it("is not thrown off by a middle trial that early-stopped on its own", () => {
    // Only the last planned trial says whether the search ran to its end: an
    // earlier one that stopped on patience is an ordinary finished trial.
    expect(
      runEnding([
        trialStart(0, 2),
        epoch(3, 10, { index: 0, of: 2 }),
        trialEnd(0, 2, 3),
        trialStart(1, 2),
        epoch(10, 10, { index: 1, of: 2 }),
        trialEnd(1, 2, 10),
        end(0, 2),
      ]),
    ).toBe("complete");
  });

  it("reads a PatchCore stopped before it scored: phases, then end, no epoch report", () => {
    expect(runEnding([start(1), phase(), end(1)])).toBe("early");
    expect(runEnding([start(1), phase(), end(0)])).toBe("early");
  });

  it("reads a PatchCore that scored as complete", () => {
    expect(runEnding([start(1), phase(), epoch(1, 1), end(1)])).toBe("complete");
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

  it("takes a unit the server marked stopped as proof, over what the stream suggests", () => {
    // The last fold's final epoch was already running when the stop landed: the
    // stream looks complete, and the server still records the fold as stopped.
    const events = [trialStart(0, 1), epoch(2, 2, { index: 0, of: 1 }), end(0, 1)];
    expect(stopOutcome({ requested: true, ended: true, events })).toBe("too-late");
    expect(
      stopOutcome({
        requested: true,
        ended: true,
        events,
        report: { fold_results: [{ status: "success" }, { status: "stopped" }] },
      }),
    ).toBe("stopped");
  });

  it("falls back to the stream when the report names no stopped unit", () => {
    expect(
      stopOutcome({
        requested: true,
        ended: true,
        events: [start(20), end(5)],
        report: { trials: [{ status: "success" }] },
      }),
    ).toBe("stopped");
    expect(
      stopOutcome({
        requested: true,
        ended: true,
        events: complete,
        report: { trials: [{ status: "success" }] },
      }),
    ).toBe("too-late");
  });

  it("never says the run finished normally without an epoch that reached its total", () => {
    expect(stopOutcome({ requested: true, ended: true, events: [] })).toBe("unknown");
    expect(
      stopOutcome({ requested: true, ended: true, events: [start(5), end(5)] }),
    ).toBe("unknown");
  });
});

describe("stopOutcome with the run's own `stopped` marker", () => {
  // The server says whether a stop cut the run (ADR-111): a field on the result,
  // not something to read off the stream or off a message.
  const complete = [epoch(4, 4), end(4)];

  it("is stopped when the server says so, whatever the stream looks like", () => {
    expect(
      stopOutcome({ requested: true, ended: true, events: complete, stopped: true }),
    ).toBe("stopped");
    expect(stopOutcome({ requested: true, ended: true, events: [], stopped: true })).toBe(
      "stopped",
    );
  });

  it("is stopped for a job stopped before any unit finished", () => {
    // The server now ends such a job as completed, with no mean and `stopped: true`.
    expect(
      stopOutcome({
        requested: true,
        ended: true,
        events: [trialStart(0, 3), epoch(1, 3, { index: 0, of: 3 }), end(0, 1)],
        report: { fold_results: [], mean_accuracy: null, stopped: true },
        stopped: true,
      }),
    ).toBe("stopped");
  });

  it("is a stop that cut nothing when the server says it did not", () => {
    // Not the stream's call: a run that ended short on its own (patience) is not
    // a stopped run either, and the server knows which it was.
    expect(
      stopOutcome({
        requested: true,
        ended: true,
        events: [epoch(3, 10), end(3)],
        stopped: false,
      }),
    ).toBe("too-late");
  });

  it("reads the stream only when the server's marker is absent (an older server)", () => {
    expect(
      stopOutcome({ requested: true, ended: true, events: [start(20), end(5)] }),
    ).toBe("stopped");
    expect(
      stopOutcome({
        requested: true,
        ended: true,
        events: [start(20), end(5)],
        stopped: undefined,
      }),
    ).toBe("stopped");
    expect(
      stopOutcome({ requested: true, ended: true, events: complete, stopped: null }),
    ).toBe("too-late");
  });

  it("never reports a stop nobody asked for, or one still closing", () => {
    expect(
      stopOutcome({ requested: false, ended: true, events: complete, stopped: true }),
    ).toBe("none");
    expect(
      stopOutcome({ requested: true, ended: false, events: [epoch(1, 4)], stopped: true }),
    ).toBe("stopping");
  });
});

describe("unitCounts", () => {
  it("counts the folds of a K-fold report", () => {
    expect(
      unitCounts({
        fold_results: [{ status: "success" }, { status: "success" }, { status: "stopped" }],
      }),
    ).toEqual({ finished: 2, stopped: 1, failed: 0, total: 3 });
  });

  it("counts the trials of a sweep, a comparison or a replicate set", () => {
    expect(
      unitCounts({
        trials: [{ status: "success" }, { status: "failed" }, { status: "stopped" }],
      }),
    ).toEqual({ finished: 1, stopped: 1, failed: 1, total: 3 });
  });

  it("reads a classification comparison, which reports counts and no list", () => {
    expect(
      unitCounts({ top_3: [], total_ran: 3, failed_count: 1, stopped_count: 1 }),
    ).toEqual({ finished: 1, stopped: 1, failed: 1, total: 3 });
  });

  it("says nothing for a report that does not list its units", () => {
    expect(unitCounts(null)).toBeNull();
    expect(unitCounts(undefined)).toBeNull();
    expect(unitCounts({})).toBeNull();
  });

  it("reads the classification grid search's own counts", () => {
    expect(
      unitCounts({
        best_trial: {},
        total_trials: 3,
        successful_trials: 1,
        failed_count: 1,
        stopped_count: 1,
        stopped: true,
      }),
    ).toEqual({ finished: 1, stopped: 1, failed: 1, total: 3 });
  });

  it("reads a grid search stopped before any trial finished: no best trial", () => {
    expect(
      unitCounts({
        best_trial: null,
        total_trials: 1,
        successful_trials: 0,
        failed_count: 0,
        stopped_count: 1,
        stopped: true,
      }),
    ).toEqual({ finished: 0, stopped: 1, failed: 0, total: 1 });
  });

  it("does not guess how the other trials ended on a report from before the counts", () => {
    // The trials that finished are known; whether the others failed or were cut is not.
    expect(
      unitCounts({ best_trial: {}, total_trials: 2, successful_trials: 1 }),
    ).toEqual({ finished: 1, stopped: null, failed: null, total: 2 });
  });

  it("does not take a grid search's unknown stopped trials for proof of a stop", () => {
    expect(
      stopOutcome({
        requested: true,
        ended: true,
        events: [trialStart(0, 3), epoch(4, 4, { index: 0, of: 3 }), trialEnd(0, 3, 4), end(0, 1)],
        report: { best_trial: {}, total_trials: 1, successful_trials: 1 },
      }),
    ).toBe("stopped");
    expect(
      stopOutcome({
        requested: true,
        ended: true,
        events: [trialStart(0, 1), epoch(4, 4, { index: 0, of: 1 }), trialEnd(0, 1, 4), end(0, 1)],
        report: { best_trial: {}, total_trials: 1, successful_trials: 1 },
      }),
    ).toBe("too-late");
  });
});

describe("stopSummary", () => {
  it("names the epoch a single run stopped at", () => {
    expect(stopSummary("epoch", [epoch(45, 400), end(45)], null)).toEqual({
      kind: "epoch",
      epoch: 45,
      total: 400,
    });
  });

  it("counts the units of a multi-unit run, from the report", () => {
    const events = [
      trialStart(0, 3),
      epoch(2, 2, { index: 0, of: 3 }),
      trialEnd(0, 3, 2),
      trialStart(1, 3),
      epoch(1, 2, { index: 1, of: 3 }),
      trialEnd(1, 3, 1),
      end(0, 2),
    ];
    expect(
      stopSummary("fold", events, {
        fold_results: [{ status: "success" }, { status: "stopped" }],
      }),
    ).toEqual({ kind: "units", unit: "fold", finished: 1, stopped: 1, planned: 3 });
  });

  it("takes the planned count from the report when the stream did not carry it", () => {
    expect(
      stopSummary("fold", [end(0, 2)], {
        n_folds: 5,
        fold_results: [{ status: "success" }, { status: "stopped" }],
      }),
    ).toEqual({ kind: "units", unit: "fold", finished: 1, stopped: 1, planned: 5 });
  });

  it("takes a standalone sweep's planned count from its report", () => {
    // The sweep report says how many trials it planned; the stream may not.
    expect(
      stopSummary("trial", [end(0, 2)], {
        planned_trials: 6,
        trials: [{ status: "success" }, { status: "stopped" }],
      }),
    ).toEqual({ kind: "units", unit: "trial", finished: 1, stopped: 1, planned: 6 });
    // The stream still wins when it has the count.
    expect(
      stopSummary("trial", [trialStart(0, 4), end(0, 2)], {
        planned_trials: 6,
        trials: [{ status: "success" }, { status: "stopped" }],
      }),
    ).toEqual({ kind: "units", unit: "trial", finished: 1, stopped: 1, planned: 4 });
  });

  it("falls back to the epoch the stream reached without a report", () => {
    expect(
      stopSummary("model", [trialStart(0, 3), epoch(1, 4, { index: 0, of: 3 }), end(0, 1)], null),
    ).toEqual({ kind: "epoch", epoch: 1, total: 4 });
  });

  it("counts a grid search's finished trials without claiming how the others ended", () => {
    const events = [
      trialStart(0, 3),
      epoch(4, 4, { index: 0, of: 3 }),
      trialEnd(0, 3, 4),
      trialStart(1, 3),
      epoch(1, 4, { index: 1, of: 3 }),
      trialEnd(1, 3, 1),
      end(0, 2),
    ];
    expect(
      stopSummary("trial", events, { best_trial: {}, total_trials: 2, successful_trials: 1 }),
    ).toEqual({ kind: "units", unit: "trial", finished: 1, stopped: null, planned: 3 });
  });

  it("takes the planned count from the submission when neither stream nor report has it", () => {
    // The classification comparison streams no events and lists only its top three.
    const report = { top_3: [], total_ran: 2, failed_count: 0, stopped_count: 1 };
    expect(stopSummary("model", [], report)).toEqual({
      kind: "units",
      unit: "model",
      finished: 1,
      stopped: 1,
      planned: null,
    });
    expect(stopSummary("model", [], report, 4)).toEqual({
      kind: "units",
      unit: "model",
      finished: 1,
      stopped: 1,
      planned: 4,
    });
  });

  it("says whether a stopped PatchCore kept its memory bank", () => {
    expect(stopSummary("phase", [start(1), phase(), end(1)], null)).toEqual({
      kind: "phase",
      bankKept: true,
    });
    expect(stopSummary("phase", [start(1), phase(), end(0)], null)).toEqual({
      kind: "phase",
      bankKept: false,
    });
  });
});

describe("stoppedFraction", () => {
  it("is the share of the planned units that finished", () => {
    expect(
      stoppedFraction("fold", { kind: "units", unit: "fold", finished: 1, stopped: 1, planned: 3 }),
    ).toBeCloseTo(1 / 3);
  });

  it("is unknown when the planned count is", () => {
    expect(
      stoppedFraction("model", { kind: "units", unit: "model", finished: 1, stopped: 1, planned: null }),
    ).toBeNull();
  });

  it("is the epoch reached of those configured, for a single run", () => {
    expect(stoppedFraction("epoch", { kind: "epoch", epoch: 45, total: 400 })).toBeCloseTo(0.1125);
    expect(stoppedFraction("epoch", { kind: "epoch", epoch: 0, total: 8 })).toBe(0);
    expect(stoppedFraction("epoch", { kind: "epoch", epoch: null, total: null })).toBeNull();
  });

  it("does not take the epoch of the trial in flight for how far a multi-unit job got", () => {
    // Trial 3 of 10 at epoch 5 of 10 is not 50% of the search.
    expect(stoppedFraction("trial", { kind: "epoch", epoch: 5, total: 10 })).toBeNull();
    expect(stoppedFraction("fold", { kind: "epoch", epoch: 1, total: 3 })).toBeNull();
  });

  it("leaves a PatchCore to its own phase fraction", () => {
    expect(stoppedFraction("phase", { kind: "phase", bankKept: true })).toBeNull();
  });
});

describe("awaitingReport", () => {
  // The hook marks the run completed, then fetches its result: for a moment the
  // run has ended and its report is not there yet. A line written in that moment
  // cannot name the units the server cut.
  it("waits for the report of a stop that has just closed", () => {
    expect(awaitingReport({ requested: true, completed: true, report: null })).toBe(true);
  });

  it("stops waiting once the report is there, even an empty one", () => {
    expect(awaitingReport({ requested: true, completed: true, report: {} })).toBe(false);
    expect(awaitingReport({ requested: true, completed: true, report: { trials: [] } })).toBe(false);
  });

  it("does not hold back a run nobody asked to stop", () => {
    expect(awaitingReport({ requested: false, completed: true, report: null })).toBe(false);
  });

  it("does not wait while the run is still going", () => {
    expect(awaitingReport({ requested: true, completed: false, report: null })).toBe(false);
  });
});

describe("isCustomJob", () => {
  it("trusts where the run came from over the queue entry", () => {
    // A custom task's sweep or replicate set is queued under a plain label, with
    // no `custom:` prefix, and the block recorded for it is a search.
    expect(isCustomJob({ task: "counting" }, "grid_search", true)).toBe(true);
    expect(isCustomJob({ task: "counting" }, "replicates", true)).toBe(true);
    expect(isCustomJob(null, "random_search", true)).toBe(true);
  });

  it("reads the queue entry when the run did not come from a custom task", () => {
    expect(isCustomJob({ task: "custom:counting" }, "grid_search", false)).toBe(true);
    expect(isCustomJob({ task: "classification" }, "custom", false)).toBe(false);
  });

  it("falls back to the block the sheet recorded", () => {
    expect(isCustomJob(null, "custom", false)).toBe(true);
    expect(isCustomJob(null, "classification", false)).toBe(false);
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

  it("confirms each stop point with what the server does there", () => {
    for (const dict of [pt, en]) {
      for (const point of POINTS) {
        expect(dict.trainingOverlay.stopConfirm[point].length, point).toBeGreaterThan(40);
      }
    }
    // A cut unit stays out of the aggregate, and the mean uses the finished ones.
    expect(pt.trainingOverlay.stopConfirm.fold).toMatch(/dobras concluídas/);
    expect(en.trainingOverlay.stopConfirm.fold).toMatch(/finished folds/);
    expect(pt.trainingOverlay.stopConfirm.model).toMatch(/ranking/);
    expect(pt.trainingOverlay.stopConfirm.replicate).toMatch(/réplicas concluídas/);
    // PatchCore: nothing is kept during extraction; the bank is finished and saved.
    expect(pt.trainingOverlay.stopConfirm.phase).toMatch(/extração/);
    expect(pt.trainingOverlay.stopConfirm.phase).toMatch(/banco/);
    expect(en.trainingOverlay.stopConfirm.phase).toMatch(/extraction/);
    expect(en.trainingOverlay.stopConfirm.phase).toMatch(/memory bank/);
    // Nothing multi-unit can be resumed.
    for (const point of ["trial", "fold", "model", "replicate"] as const) {
      expect(pt.trainingOverlay.stopConfirm[point], point).toMatch(/retomad/);
      expect(en.trainingOverlay.stopConfirm[point], point).toMatch(/resum/);
    }
  });

  it("keeps the confirmation for a custom task to what the server does for it", () => {
    for (const dict of [pt, en]) {
      for (const point of ["epoch", "trial", "replicate"] as const) {
        expect(dict.trainingOverlay.stopConfirmCustom[point].length, point).toBeGreaterThan(40);
      }
    }
    // A custom task is never resumable from the History.
    expect(pt.trainingOverlay.stopConfirmCustom.epoch).not.toMatch(/retomar pelo Histórico/);
    expect(en.trainingOverlay.stopConfirmCustom.epoch).not.toMatch(/resume it from the History/);
    // A task that owns its training loop runs the trial or replicate in flight to the end.
    expect(pt.trainingOverlay.stopConfirmCustom.trial).toMatch(/até o fim/);
    expect(pt.trainingOverlay.stopConfirmCustom.replicate).toMatch(/até o fim/);
    expect(en.trainingOverlay.stopConfirmCustom.trial).toMatch(/to the end/);
    expect(en.trainingOverlay.stopConfirmCustom.replicate).toMatch(/to the end/);
  });

  it("counts the finished units against the planned ones, and the one left out", () => {
    expect(pt.trainingOverlay.stoppedUnitsLog("fold", 2, 3, 1)).toMatch(/dobras concluídas: 2\/3/);
    expect(pt.trainingOverlay.stoppedUnitsLog("fold", 2, 3, 1)).toMatch(/fora da agregação: 1/);
    expect(en.trainingOverlay.stoppedUnitsLog("model", 1, 4, 1)).toMatch(/models finished: 1\/4/);
    // No stopped unit, no mention; no planned count, no "/null".
    expect(en.trainingOverlay.stoppedUnitsLog("trial", 2, 5, 0)).not.toMatch(/left out/);
    expect(pt.trainingOverlay.stoppedUnitsLog("replicate", 2, null, 0)).not.toMatch(/null|\//);
    expect(en.trainingOverlay.stoppedUnitsLog("replicate", 2, null, 0)).not.toMatch(/null|\//);
    // A report that cannot tell how the unfinished ones ended says nothing about them.
    expect(pt.trainingOverlay.stoppedUnitsLog("trial", 1, 3, null)).toBe(
      "interrompido · trials concluídos: 1/3",
    );
    expect(en.trainingOverlay.stoppedUnitsLog("trial", 1, 3, null)).toBe(
      "stopped · trials finished: 1/3",
    );
  });

  it("says whether the stopped PatchCore kept its memory bank", () => {
    expect(pt.trainingOverlay.stoppedPhaseLog(true)).toMatch(/banco/);
    expect(pt.trainingOverlay.stoppedPhaseLog(false)).toMatch(/nada/);
    expect(en.trainingOverlay.stoppedPhaseLog(true)).toMatch(/memory bank/);
    expect(en.trainingOverlay.stoppedPhaseLog(false)).toMatch(/nothing/);
  });

  it("says a stop that cut nothing cut nothing, not that it came in the last epoch", () => {
    // The server's marker also says false for a run that finished early on its own.
    expect(pt.trainingOverlay.stopTooLate).not.toMatch(/última época/);
    expect(en.trainingOverlay.stopTooLate).not.toMatch(/last epoch/);
  });

  it("speaks of the run, not of its kind, in the right gender", () => {
    // "Esta execução … interrompida", not "Este tipo de execução … interrompida".
    expect(pt.trainingOverlay.stopUnavailable).toMatch(/^Esta execução/);
    expect(pt.queueOverlay.stopUnavailable).toMatch(/^Esta execução/);
  });
});
