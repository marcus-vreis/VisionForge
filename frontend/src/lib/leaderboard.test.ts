import { describe, expect, it } from "vitest";

import type { MetricAggregate, RunGroupBrief, RunSummary } from "../types/run";
import {
  boardMetrics,
  datasetIdentity,
  defaultMetric,
  groupBoards,
  intervalsOverlap,
  normalizeDatasetPath,
  rankEntries,
  rankingCaution,
} from "./leaderboard";
import { foldGroups, type HistoryEntry } from "./run-groups";

function run(id: string, extra: Partial<RunSummary> = {}): RunSummary {
  return {
    run_id: id,
    experiment_name: id,
    model_arch: "resnet18",
    task: "multiclass",
    status: "completed",
    started_at: "2026-05-22T10:00:00",
    finished_at: "2026-05-22T10:05:00",
    epochs_completed: 5,
    final_metrics: { accuracy: 0.9, f1: 0.88, val_loss: 0.3 },
    dataset_name: "coffee",
    dataset_root: "C:/data/coffee",
    ...extra,
  };
}

function entry(r: RunSummary, children: RunSummary[] = []): HistoryEntry {
  return { run: r, children };
}

function aggregate(mean: number, low: number | null, high: number | null, n: number): MetricAggregate {
  return {
    n,
    mean,
    std: n > 1 ? 0.01 : null,
    min: mean,
    max: mean,
    ci95_low: low,
    ci95_high: high,
  } as MetricAggregate;
}

function group(
  id: string,
  metric: string,
  agg: MetricAggregate,
  extra: Partial<RunGroupBrief> = {},
  summary: Partial<RunSummary> = {},
): RunSummary {
  return run(id, {
    final_metrics: { [metric]: agg.mean as number },
    group: {
      kind: "replicates",
      metric,
      seeds: [1, 2, 3],
      n_requested: agg.n,
      n_finished: agg.n,
      stopped: false,
      child_ids: [],
      final_aggregates: { [metric]: agg },
      variants: [],
      best_by_mean: null,
      ...extra,
    },
    ...summary,
  });
}

const ids = (rows: ReadonlyArray<{ entry: HistoryEntry }>) => rows.map((r) => r.entry.run.run_id);

describe("normalizeDatasetPath", () => {
  it("reads one folder spelled several ways as one", () => {
    const spellings = ["C:\\data\\Coffee\\", "c:/data/coffee", "C:/data//coffee/", " c:\\DATA\\coffee "];
    expect(new Set(spellings.map(normalizeDatasetPath)).size).toBe(1);
  });

  it("keeps the case of a path that is not on a Windows drive", () => {
    expect(normalizeDatasetPath("/data/Coffee")).not.toBe(normalizeDatasetPath("/data/coffee"));
  });

  it("keeps the leading slashes of a network share", () => {
    expect(normalizeDatasetPath("\\\\server\\share\\coffee\\")).toBe("//server/share/coffee");
  });
});

describe("datasetIdentity", () => {
  it("prefers the fingerprint, and keeps the method in the key", () => {
    const a = datasetIdentity({ dataset_digest: "abc", dataset_method: "manifest", dataset_root: "x" });
    const b = datasetIdentity({ dataset_digest: "abc", dataset_method: "content", dataset_root: "x" });
    expect(a.basis).toBe("fingerprint");
    expect(a.key).not.toBe(b.key);
  });

  it("separates one path holding two different fingerprints", () => {
    const a = datasetIdentity({ dataset_digest: "aaa", dataset_method: "manifest", dataset_root: "C:/d" });
    const b = datasetIdentity({ dataset_digest: "bbb", dataset_method: "manifest", dataset_root: "C:/d" });
    expect(a.key).not.toBe(b.key);
  });

  it("falls back to the normalized path when there is no digest", () => {
    const a = datasetIdentity({ dataset_digest: null, dataset_root: "C:\\data\\coffee\\" });
    const b = datasetIdentity({ dataset_root: "c:/data/coffee" });
    expect(a).toEqual({ key: "path:c:/data/coffee", basis: "path" });
    expect(b.key).toBe(a.key);
  });

  it("calls a run with neither unknown", () => {
    expect(datasetIdentity({ dataset_root: null }).basis).toBe("unknown");
    expect(datasetIdentity({ dataset_root: "  " }).basis).toBe("unknown");
    expect(datasetIdentity({}).key).toBe("unknown");
  });
});

describe("groupBoards", () => {
  it("makes one board per task and dataset", () => {
    const boards = groupBoards([
      entry(run("a")),
      entry(run("b")),
      entry(run("c", { dataset_root: "C:/data/other", dataset_name: "other" })),
      entry(run("d", { task: "binary" })),
      entry(run("e", { task: "detection", final_metrics: { map50: 0.5 } })),
    ]);
    expect(boards.map((b) => ids(b.entries.map((e) => ({ entry: e }))))).toEqual([
      ["a", "b"],
      ["c"],
      ["d"],
      ["e"],
    ]);
  });

  it("does not merge a path-only run into a fingerprinted board with the same path", () => {
    const boards = groupBoards([
      entry(run("fp", { dataset_digest: "abc", dataset_method: "manifest" })),
      entry(run("old")),
    ]);
    expect(boards).toHaveLength(2);
    expect(boards.map((b) => b.identity.basis)).toEqual(["fingerprint", "path"]);
  });

  it("puts runs with no dataset in a board of their own, last", () => {
    const boards = groupBoards([
      entry(run("none", { dataset_root: null, dataset_name: null })),
      entry(run("a")),
    ]);
    expect(boards.map((b) => b.identity.basis)).toEqual(["path", "unknown"]);
  });

  it("lists a replicate group once, its seeds staying under it", () => {
    const seed = run("seed1", { group_id: "g" });
    const g = group("g", "accuracy", aggregate(0.9, 0.88, 0.92, 3));
    const boards = groupBoards(foldGroups([seed, g]));
    expect(boards).toHaveLength(1);
    expect(boards[0].entries.map((e) => e.run.run_id)).toEqual(["g"]);
  });
});

describe("boardMetrics and defaultMetric", () => {
  it("opens classification on the held-out accuracy, not on the validation loss", () => {
    const metrics = boardMetrics([entry(run("a"))]);
    expect(metrics.map((m) => m.key)).toEqual(["val_loss", "accuracy", "f1"]);
    expect(defaultMetric(metrics)).toBe("accuracy");
  });

  it("opens each task on the first quality row of its table", () => {
    const cases: Array<[string, Record<string, number>, string]> = [
      ["detection", { map50: 0.6, map50_95: 0.4 }, "map50_95"],
      ["regression", { mae: 3, rmse: 4, r2: 0.7 }, "r2"],
      ["segmentation", { pixel_acc: 0.9, dice: 0.8, miou: 0.7 }, "miou"],
      ["anomaly", { f1: 0.5, auroc: 0.9 }, "auroc"],
    ];
    for (const [task, final_metrics, expected] of cases) {
      const metrics = boardMetrics([entry(run("a", { task, final_metrics }))]);
      expect(defaultMetric(metrics), task).toBe(expected);
    }
  });

  it("keeps the validation fallback apart from the held-out score", () => {
    const metrics = boardMetrics([
      entry(run("a")),
      entry(run("b", { final_metrics: { val_accuracy: 0.8, val_f1: 0.7, val_loss: 0.4 } })),
    ]);
    const keys = metrics.map((m) => m.key);
    expect(keys).toContain("accuracy");
    expect(keys).toContain("val_accuracy");
    expect(defaultMetric(metrics)).toBe("accuracy");
  });

  it("falls back to the loss when it is all the runs reported", () => {
    const metrics = boardMetrics([entry(run("a", { final_metrics: { val_loss: 0.3 } }))]);
    expect(defaultMetric(metrics)).toBe("val_loss");
  });

  it("takes a researcher's task in the order it reported", () => {
    const metrics = boardMetrics([
      entry(run("a", { task: "custom:shapes", final_metrics: { score: 0.3, iou: 0.7 } })),
    ]);
    expect(metrics.map((m) => m.key)).toEqual(["score", "iou"]);
    expect(defaultMetric(metrics)).toBe("score");
  });

  it("reads the direction from the server, else from the name", () => {
    const served = boardMetrics([
      entry(
        run("a", {
          task: "custom:shapes",
          final_metrics: { score: 0.3, iou: 0.7 },
          metric_directions: { score: "lower", iou: "higher" },
        }),
      ),
    ]);
    expect(Object.fromEntries(served.map((m) => [m.key, m.direction]))).toEqual({
      score: "lower",
      iou: "higher",
    });

    const inferred = boardMetrics([entry(run("a"))]);
    expect(Object.fromEntries(inferred.map((m) => [m.key, m.direction]))).toEqual({
      val_loss: "lower",
      accuracy: "higher",
      f1: "higher",
    });
  });

  it("is empty for a board whose runs measured nothing", () => {
    expect(defaultMetric(boardMetrics([entry(run("a", { final_metrics: {} }))]))).toBeNull();
    expect(boardMetrics([])).toEqual([]);
  });
});

describe("rankEntries", () => {
  it("orders a higher-is-better metric from the largest down", () => {
    const { ranked } = rankEntries(
      [
        entry(run("low", { final_metrics: { accuracy: 0.7 } })),
        entry(run("high", { final_metrics: { accuracy: 0.9 } })),
        entry(run("mid", { final_metrics: { accuracy: 0.8 } })),
      ],
      "accuracy",
      "higher",
    );
    expect(ids(ranked)).toEqual(["high", "mid", "low"]);
    expect(ranked.map((r) => r.rank)).toEqual([1, 2, 3]);
  });

  it("orders a lower-is-better metric from the smallest up", () => {
    const { ranked } = rankEntries(
      [
        entry(run("a", { final_metrics: { val_loss: 0.5 } })),
        entry(run("b", { final_metrics: { val_loss: 0.2 } })),
      ],
      "val_loss",
      "lower",
    );
    expect(ids(ranked)).toEqual(["b", "a"]);
  });

  it("gives equal values the same rank", () => {
    const { ranked } = rankEntries(
      [
        entry(run("a", { final_metrics: { accuracy: 0.9 } })),
        entry(run("b", { final_metrics: { accuracy: 0.9 } })),
        entry(run("c", { final_metrics: { accuracy: 0.8 } })),
      ],
      "accuracy",
      "higher",
    );
    expect(ranked.map((r) => r.rank)).toEqual([1, 1, 3]);
  });

  it("ranks a replicate group by its mean and carries the interval it came with", () => {
    const g = group("g", "accuracy", aggregate(0.91, 0.89, 0.93, 5));
    const { ranked } = rankEntries([entry(run("single")), entry(g)], "accuracy", "higher");
    expect(ids(ranked)).toEqual(["g", "single"]);
    expect(ranked[0]).toMatchObject({ seeds: 5, basis: "mean", low: 0.89, high: 0.93 });
    expect(ranked[1]).toMatchObject({ seeds: 1, basis: "value", low: null, high: null });
  });

  it("reads a group with one finished seed as the single seed it is", () => {
    const g = group("g", "accuracy", aggregate(0.91, null, null, 1));
    const { ranked } = rankEntries([entry(g)], "accuracy", "higher");
    expect(ranked[0]).toMatchObject({ seeds: 1, basis: "value", low: null, high: null });
  });

  it("lists after the ranked, with the reason, what cannot be ranked", () => {
    const stopped = run("stopped", { stopped: true });
    const groupStopped = group("gs", "accuracy", aggregate(0.99, 0.98, 1, 3), { stopped: true });
    const noMetric = run("none", { final_metrics: { f1: 0.5 } });
    const failed = run("failed", { status: "failed" });
    const variants = run("variants", {
      final_metrics: {},
      group: {
        kind: "replicated_comparison",
        metric: "accuracy",
        seeds: [1, 2],
        n_requested: 4,
        n_finished: 4,
        stopped: false,
        child_ids: [],
        final_aggregates: {},
        variants: ["a", "b"],
        best_by_mean: "a",
      },
    });
    const ok = run("ok", { final_metrics: { accuracy: 0.5 } });

    const { ranked, unranked } = rankEntries(
      [entry(stopped), entry(groupStopped), entry(noMetric), entry(failed), entry(variants), entry(ok)],
      "accuracy",
      "higher",
    );

    expect(ids(ranked)).toEqual(["ok"]);
    expect(unranked.map((u) => [u.entry.run.run_id, u.reason])).toEqual([
      ["stopped", "stopped"],
      ["gs", "stopped"],
      ["none", "no-metric"],
      ["failed", "not-finished"],
      ["variants", "comparison"],
    ]);
  });

  it("does not rank a zero away", () => {
    const { ranked, unranked } = rankEntries(
      [entry(run("zero", { final_metrics: { accuracy: 0 } }))],
      "accuracy",
      "higher",
    );
    expect(ids(ranked)).toEqual(["zero"]);
    expect(unranked).toEqual([]);
  });
});

describe("intervalsOverlap", () => {
  it("counts a shared endpoint as overlap", () => {
    expect(intervalsOverlap({ low: 0.1, high: 0.5 }, { low: 0.5, high: 0.9 })).toBe(true);
    expect(intervalsOverlap({ low: 0.1, high: 0.4 }, { low: 0.5, high: 0.9 })).toBe(false);
    expect(intervalsOverlap({ low: 0.1, high: 0.9 }, { low: 0.4, high: 0.5 })).toBe(true);
  });
});

describe("rankingCaution", () => {
  const rank = (...entries: HistoryEntry[]) => rankEntries(entries, "accuracy", "higher");
  const single = (id: string, value: number) =>
    entry(run(id, { final_metrics: { accuracy: value } }));
  const mean = (id: string, value: number, low: number, high: number) =>
    entry(group(id, "accuracy", aggregate(value, low, high, 5)));

  it("says nothing with fewer than two ranked runs", () => {
    expect(rank().caution).toBeNull();
    expect(rank(single("a", 0.9)).caution).toBeNull();
  });

  it("warns when the top two are single seeds", () => {
    expect(rank(single("a", 0.9), single("b", 0.8), single("c", 0.7)).caution).toBe("single-seed");
  });

  it("warns when only one of the top two is a single seed", () => {
    expect(rank(mean("g", 0.9, 0.88, 0.92), single("b", 0.8)).caution).toBe("single-seed-mixed");
  });

  it("judges the top two only: a single seed further down does not matter", () => {
    const { caution } = rank(
      mean("g1", 0.95, 0.94, 0.96),
      mean("g2", 0.8, 0.78, 0.82),
      single("c", 0.7),
    );
    expect(caution).toBeNull();
  });

  it("warns when the intervals of the top two overlap", () => {
    expect(rank(mean("g1", 0.9, 0.86, 0.94), mean("g2", 0.88, 0.84, 0.92)).caution).toBe("overlap");
  });

  it("stays silent when the intervals are apart", () => {
    expect(rank(mean("g1", 0.95, 0.94, 0.96), mean("g2", 0.8, 0.78, 0.82)).caution).toBeNull();
  });

  it("warns that it cannot judge when a mean carries no interval", () => {
    const noInterval = entry(group("g2", "accuracy", aggregate(0.88, null, null, 5)));
    expect(rank(mean("g1", 0.9, 0.86, 0.94), noInterval).caution).toBe("no-interval");
  });

  it("is a pure function of the rows it is given", () => {
    const { ranked } = rank(single("a", 0.9), single("b", 0.8));
    expect(rankingCaution(ranked)).toBe("single-seed");
    expect(rankingCaution(ranked.slice(0, 1))).toBeNull();
  });
});
