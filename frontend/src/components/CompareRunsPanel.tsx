import { useEffect, useState } from "react";
import { fetchRunDetail, fetchTasks, type RunDetail } from "../api/client";
import { useT } from "../i18n/useT";
import type { Dict } from "../i18n/pt";
import {
  distinctTasks,
  extremeIndexes,
  isCustomTaskKey,
  metricRows,
  numericMetric,
  runTaskKey,
} from "../lib/compare-metrics";
import { curveSeries, selectedCurves, toggleCurve, type CurveSeries } from "../lib/compare-curves";
import { seedNote } from "../lib/compare-seeds";
import type { TaskDescriptor } from "../lib/custom-tasks";
import { compareDatasets } from "../lib/dataset-identity";
import { aggregateForRow, formatAggregate } from "../lib/run-groups";
import { accentForTask } from "../lib/task-accent";

interface CompareRunsPanelProps {
  runIds: string[];
  onBack: () => void;
}

const PALETTE = [
  "oklch(0.78 0.18 150)", // green
  "oklch(0.74 0.18 22)", // red
  "oklch(0.74 0.16 240)", // blue
  "oklch(0.78 0.18 305)", // purple
  "oklch(0.84 0.18 75)", // amber
  "oklch(0.78 0.16 200)", // teal
];

function fmtMetric(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") {
    // A never-measured metric is null; an infinity or NaN must not be printed.
    if (!Number.isFinite(v)) return "—";
    return v % 1 === 0 ? String(v) : v.toFixed(4);
  }
  return String(v);
}

/** Side-by-side metric + overlaid epoch-curve view for 2+ historical runs. */
export function CompareRunsPanel({ runIds, onBack }: CompareRunsPanelProps) {
  const t = useT();
  const [details, setDetails] = useState<RunDetail[]>([]);
  // Only read for researcher-defined tasks: what they declared about their metrics.
  const [descriptors, setDescriptors] = useState<TaskDescriptor[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    // Defer the loading flag out of the synchronous effect body; cleared in
    // finally so a fast resolve never leaves it stuck on.
    const loadingTimer = setTimeout(() => {
      if (alive) setLoading(true);
    }, 0);
    Promise.all(runIds.map((id) => fetchRunDetail(id)))
      .then(async (arr) => {
        // A comparison does not depend on the descriptors: without them a custom
        // task's metrics lose their declared direction (the name decides) and
        // its label, and nothing else.
        const custom = arr.some((d) => isCustomTaskKey(runTaskKey(d)));
        const known = custom
          ? await fetchTasks()
              .then((r) => r.tasks)
              .catch((): TaskDescriptor[] => [])
          : [];
        if (alive) {
          setDetails(arr);
          setDescriptors(known);
        }
      })
      .catch((e: unknown) => {
        if (alive) {
          setError(e instanceof Error ? e.message : t.compareRuns.loadFailed);
        }
      })
      .finally(() => {
        clearTimeout(loadingTimer);
        if (alive) setLoading(false);
      });
    return () => {
      alive = false;
      clearTimeout(loadingTimer);
    };
  }, [runIds.join(",")]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <button
          type="button"
          onClick={onBack}
          style={{
            padding: "6px 12px",
            background: "rgba(255,255,255,0.04)",
            border: "1px solid var(--vf-panel-stroke)",
            borderRadius: 8,
            color: "var(--vf-text-dim)",
            fontFamily: "var(--font-mono)",
            fontSize: 11,
            letterSpacing: "0.10em",
            textTransform: "uppercase",
            cursor: "pointer",
          }}
        >
          {t.compareRuns.back}
        </button>
        <div style={{ fontFamily: "var(--font-mono)", fontSize: 14, color: "var(--vf-text)" }}>
          {t.compareRuns.comparing(runIds.length)}
        </div>
      </div>

      {loading && (
        <div style={{ padding: 32, textAlign: "center", color: "var(--vf-text-muted)" }}>
          {t.compareRuns.loading}
        </div>
      )}

      {error && (
        <div
          style={{
            padding: 14,
            background: "oklch(0.704 0.191 22.216 / 0.10)",
            border: "1px solid oklch(0.704 0.191 22.216 / 0.4)",
            borderRadius: 10,
            color: "oklch(0.85 0.14 22)",
            fontFamily: "var(--font-mono)",
            fontSize: 12,
          }}
        >
          {error}
        </div>
      )}

      {!loading && !error && distinctTasks(details).length > 1 && (
        <MixedTasks details={details} descriptors={descriptors} />
      )}

      {!loading && !error && details.length > 0 && distinctTasks(details).length === 1 && (
        <>
          <Legend details={details} />
          <DatasetVerdictRow details={details} />
          <MetricsTable details={details} descriptors={descriptors} />
          <ConfigDiffTable details={details} />
          <PreprocessingCompare details={details} />
          <EpochCurves details={details} task={distinctTasks(details)[0]} />
        </>
      )}
    </div>
  );
}

const VERDICT_STYLE = {
  same: { icon: "✓", color: "oklch(0.80 0.15 150)" },
  different: { icon: "✗", color: "oklch(0.72 0.19 25)" },
  unknown: { icon: "⚠", color: "oklch(0.80 0.13 85)" },
};

/** Whether the runs being compared saw the same data.
 *
 * Comparing metrics across different datasets is the mistake the fingerprint
 * exists to prevent, so the answer belongs next to the metrics. Every run is
 * compared against the first, and the weakest verdict wins: one unanswerable
 * pair makes the whole set unanswerable, because a "same" that skipped a run
 * would overclaim.
 */
function DatasetVerdictRow({ details }: { details: RunDetail[] }) {
  const t = useT();
  if (details.length < 2) return null;

  const verdicts = details.slice(1).map((d) => compareDatasets(t, details[0].dataset, d.dataset));
  const verdict =
    verdicts.find((v) => v.kind === "unknown") ??
    verdicts.find((v) => v.kind === "different") ??
    verdicts[0];
  const style = VERDICT_STYLE[verdict.kind];
  const names = Array.from(new Set(details.map((d) => d.dataset?.name).filter(Boolean)));

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 8,
        padding: "8px 12px",
        background: "rgba(255,255,255,0.02)",
        border: "1px solid var(--vf-panel-stroke)",
        borderRadius: 10,
        fontFamily: "var(--font-mono)",
        fontSize: 11,
        color: style.color,
      }}
    >
      <span>{style.icon}</span>
      <span>
        {t.compareRuns.verdict[verdict.kind]}
        {verdict.kind === "unknown" ? ` — ${verdict.reason}` : ""}
      </span>
      {names.length > 0 && (
        <span style={{ color: "var(--vf-text-muted)" }}>· 🗂 {names.join(", ")}</span>
      )}
    </div>
  );
}

function Legend({ details }: { details: RunDetail[] }) {
  const t = useT();
  return (
    <div
      style={{
        display: "flex",
        flexWrap: "wrap",
        gap: 10,
        padding: "8px 12px",
        background: "rgba(255,255,255,0.02)",
        border: "1px solid var(--vf-panel-stroke)",
        borderRadius: 10,
      }}
    >
      {details.map((d, i) => (
        <div
          key={d.run_id}
          style={{
            display: "flex",
            alignItems: "center",
            gap: 6,
            fontFamily: "var(--font-mono)",
            fontSize: 11,
            color: "var(--vf-text-dim)",
          }}
        >
          <span
            style={{
              width: 10,
              height: 10,
              borderRadius: 3,
              background: PALETTE[i % PALETTE.length],
              boxShadow: `0 0 6px ${PALETTE[i % PALETTE.length]}`,
            }}
          />
          {d.experiment_name}
          {d.group && (
            <span style={{ color: "var(--vf-text-muted)" }}>
              {" · "}
              {t.runGroup.kind[d.group.kind]}
            </span>
          )}
        </div>
      ))}
    </div>
  );
}

/** The name a task goes by on screen: the dictionary's for a built-in, the
 *  researcher's own label (else the key) for a custom task. */
function taskDisplayName(t: Dict, task: string, descriptors: TaskDescriptor[]): string {
  if (isCustomTaskKey(task)) {
    const key = task.slice("custom:".length);
    return descriptors.find((d) => d.key === key)?.label ?? key;
  }
  const names: Record<string, string> = t.taskNames;
  return names[task] ?? task;
}

/** Why there is no comparison: runs of different tasks have no metric in common
 *  to set side by side, so the panel says so and lists whose is whose. */
function MixedTasks({
  details,
  descriptors,
}: {
  details: RunDetail[];
  descriptors: TaskDescriptor[];
}) {
  const t = useT();
  return (
    <div
      style={{
        padding: 14,
        background: "oklch(0.80 0.13 85 / 0.08)",
        border: "1px solid oklch(0.80 0.13 85 / 0.4)",
        borderRadius: 12,
        fontFamily: "var(--font-mono)",
        fontSize: 12,
        color: "var(--vf-text)",
        display: "flex",
        flexDirection: "column",
        gap: 8,
      }}
    >
      <div style={{ color: "oklch(0.88 0.13 85)", fontWeight: 600 }}>
        ⚠ {t.compareRuns.mixedTasks.title}
      </div>
      <div style={{ color: "var(--vf-text-dim)", lineHeight: 1.5 }}>
        {t.compareRuns.mixedTasks.body}
      </div>
      <ul style={{ margin: 0, paddingLeft: 18, color: "var(--vf-text-muted)" }}>
        {details.map((d) => (
          <li key={d.run_id}>
            {d.experiment_name} ·{" "}
            <strong>{taskDisplayName(t, runTaskKey(d), descriptors)}</strong>
          </li>
        ))}
      </ul>
    </div>
  );
}

/** The glyph beside a metric's name that says which end of it is better. */
const DIRECTION_GLYPH = { higher: "↑", lower: "↓" };

function MetricsTable({
  details,
  descriptors,
}: {
  details: RunDetail[];
  descriptors: TaskDescriptor[];
}) {
  const t = useT();
  const task = runTaskKey(details[0]);
  const declared = isCustomTaskKey(task)
    ? descriptors.find((d) => d.key === task.slice("custom:".length))?.metrics
    : undefined;
  const rows = metricRows(task, details, declared).map((row) => ({
    row,
    values: details.map((d) => numericMetric(d.metrics[row.key])),
  }));
  // A group's cell is a mean over seeds with its own interval, but the extreme
  // the table highlights is judged over every cell, group means included. The
  // single-seed caution (ADR-112) therefore speaks whenever a highlighted row
  // has a single-seed run on it (ADR-113), in wording that names which runs.
  const isGroup = details.map((d) => d.group != null);
  const note = seedNote(
    rows.map(({ row, values }) => ({ direction: row.direction, values })),
    isGroup,
  );
  return (
    <div
      style={{
        padding: 14,
        background: "rgba(255,255,255,0.025)",
        border: "1px solid var(--vf-panel-stroke)",
        borderRadius: 12,
        overflowX: "auto",
      }}
    >
      <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: "var(--font-mono)", fontSize: 12 }}>
        <thead>
          <tr>
            <th style={thStyle}>{t.compareRuns.metric}</th>
            {details.map((d, i) => (
              <th key={d.run_id} style={{ ...thStyle, color: PALETTE[i % PALETTE.length] }}>
                {d.experiment_name}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map(({ row, values }) => {
            const label = row.label ? t.compareRuns.metrics[row.label] : row.key;
            const extremes = extremeIndexes(values, row.direction);
            return (
              <tr key={row.key}>
                <td style={tdLabelStyle}>
                  {label}
                  {row.direction && (
                    <span
                      title={t.compareRuns.direction[row.direction]}
                      style={{ marginLeft: 6, opacity: 0.6, cursor: "help" }}
                    >
                      {DIRECTION_GLYPH[row.direction]}
                    </span>
                  )}
                </td>
                {details.map((d, i) => {
                  // The highest or lowest value of the row, said as that and
                  // nothing more: whether it is a real difference is not
                  // something one run each can tell (ADR-112).
                  const extreme = extremes.includes(i) && row.direction;
                  // A group's cell is its mean with the interval the report gave
                  // (never one seed's value); a replicated comparison has no
                  // single number per row, so it prints a dash and says why.
                  const shown = d.group ? formatAggregate(aggregateForRow(d.group, row.key)) : null;
                  const titles = [
                    shown
                      ? t.compareRuns.group.cellTitle(shown.n, shown.low, shown.high)
                      : d.group?.kind === "replicated_comparison"
                        ? t.compareRuns.group.noCell
                        : null,
                    extreme
                      ? t.compareRuns.extreme[extreme === "higher" ? "highest" : "lowest"]
                      : null,
                  ].filter((title): title is string => title !== null);
                  return (
                    <td
                      key={d.run_id}
                      title={titles.length > 0 ? titles.join(" · ") : undefined}
                      style={extreme ? { ...tdStyle, ...extremeStyle } : tdStyle}
                    >
                      {d.group ? (
                        shown ? (
                          <>
                            {shown.mean}
                            {shown.half !== null && (
                              <span style={{ color: "var(--vf-text-muted)", fontWeight: 400 }}>
                                {" ± "}
                                {shown.half}
                              </span>
                            )}
                          </>
                        ) : (
                          "—"
                        )
                      ) : (
                        fmtMetric(d.metrics[row.key])
                      )}
                    </td>
                  );
                })}
              </tr>
            );
          })}
          <tr>
            <td style={tdLabelStyle}>{t.compareRuns.device}</td>
            {details.map((d) => (
              <td key={d.run_id} style={tdStyle}>
                {d.device_used ?? "—"}
              </td>
            ))}
          </tr>
        </tbody>
      </table>
      {note && (
        <div
          role="note"
          style={{
            marginTop: 10,
            paddingTop: 10,
            borderTop: "1px solid rgba(255,255,255,0.04)",
            fontFamily: "var(--font-mono)",
            fontSize: 11,
            lineHeight: 1.5,
            color: "oklch(0.86 0.12 85)",
          }}
        >
          ⚠ {t.compareRuns.seedNote[note]}
        </div>
      )}
    </div>
  );
}

const thStyle: React.CSSProperties = {
  textAlign: "left",
  padding: "8px 10px",
  borderBottom: "1px solid var(--vf-panel-stroke)",
  fontSize: 10,
  letterSpacing: "0.14em",
  textTransform: "uppercase",
  color: "var(--vf-text-muted)",
  fontWeight: 500,
};

const tdStyle: React.CSSProperties = {
  padding: "8px 10px",
  borderBottom: "1px solid rgba(255,255,255,0.04)",
  color: "var(--vf-text)",
};

/** The cell holding a row's highest or lowest value. Neutral on purpose: no
 *  green, no "best" — see `extremeIndexes`. */
const extremeStyle: React.CSSProperties = {
  background: "rgba(255,255,255,0.07)",
  fontWeight: 700,
};

const tdLabelStyle: React.CSSProperties = {
  ...tdStyle,
  color: "var(--vf-text-muted)",
  fontSize: 11,
  letterSpacing: "0.04em",
};

// ── Config diff ──────────────────────────────────────────────────────────────

/** Selectors that pull comparable scalar values from RunDetail.config. */
const CONFIG_ROWS: Array<{ id: keyof Dict["compareRuns"]["config"]; pick: (cfg: Record<string, unknown>) => unknown }> = [
  { id: "architecture", pick: (c) => (c.model as Record<string, unknown> | undefined)?.name },
  { id: "numClasses", pick: (c) => (c.model as Record<string, unknown> | undefined)?.num_classes },
  { id: "pretrained", pick: (c) => (c.model as Record<string, unknown> | undefined)?.pretrained },
  { id: "task", pick: (c) => c.task },
  { id: "learningRate", pick: (c) => (c.training as Record<string, unknown> | undefined)?.learning_rate },
  { id: "optimizer", pick: (c) => (c.training as Record<string, unknown> | undefined)?.optimizer },
  { id: "batchSize", pick: (c) => (c.training as Record<string, unknown> | undefined)?.batch_size },
  { id: "epochsMax", pick: (c) => (c.training as Record<string, unknown> | undefined)?.epochs },
  { id: "weightDecay", pick: (c) => (c.training as Record<string, unknown> | undefined)?.weight_decay },
  { id: "seed", pick: (c) => (c.training as Record<string, unknown> | undefined)?.seed },
  { id: "mixedPrecision", pick: (c) => (c.training as Record<string, unknown> | undefined)?.mixed_precision },
  {
    id: "scheduler",
    pick: (c) => {
      const s = (c.training as Record<string, unknown> | undefined)?.scheduler as
        | Record<string, unknown>
        | undefined;
      return s?.kind ?? "none";
    },
  },
  { id: "imageSize", pick: (c) => pickDataTransforms(c)?.image_size },
  { id: "horizontalFlip", pick: (c) => pickDataTransforms(c)?.horizontal_flip },
  { id: "rotation", pick: (c) => pickDataTransforms(c)?.rotation_degrees },
  { id: "colorJitter", pick: (c) => pickDataTransforms(c)?.color_jitter },
  {
    id: "preprocessing",
    pick: (c) => {
      const steps = pickPreprocessingSteps(c);
      if (steps.length === 0) return "—";
      return steps.map((s) => String(s.kind ?? "")).join(" → ");
    },
  },
];

function pickDataTransforms(
  config: Record<string, unknown>,
): Record<string, unknown> | undefined {
  const data = config.data as Record<string, unknown> | undefined;
  return data?.transforms as Record<string, unknown> | undefined;
}

function pickPreprocessingSteps(
  config: Record<string, unknown>,
): Array<Record<string, unknown>> {
  const data = config.data as Record<string, unknown> | undefined;
  const pp = data?.preprocessing as Record<string, unknown> | undefined;
  const steps = pp?.steps;
  return Array.isArray(steps) ? (steps as Array<Record<string, unknown>>) : [];
}

function fmtConfigValue(v: unknown): string {
  if (v === null || v === undefined || v === "") return "—";
  if (v === true) return "✓";
  if (v === false) return "—";
  if (typeof v === "number") return v % 1 === 0 ? String(v) : String(v);
  return String(v);
}

/** Side-by-side hyperparameter comparison with diff highlighting.
 *
 * Highlights cells where the run's value differs from the first run — lets
 * the researcher attribute metric deltas to specific config changes instead
 * of guessing why one curve beats another.
 */
function ConfigDiffTable({ details }: { details: RunDetail[] }) {
  const t = useT();
  return (
    <div
      style={{
        padding: 14,
        background: "rgba(255,255,255,0.025)",
        border: "1px solid var(--vf-panel-stroke)",
        borderRadius: 12,
        overflowX: "auto",
      }}
    >
      <div
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: 10,
          letterSpacing: "0.16em",
          textTransform: "uppercase",
          color: "var(--vf-text-muted)",
          marginBottom: 10,
        }}
      >
        {t.compareRuns.configDiffTitle}
      </div>
      <table
        style={{
          width: "100%",
          borderCollapse: "collapse",
          fontFamily: "var(--font-mono)",
          fontSize: 12,
        }}
      >
        <thead>
          <tr>
            <th style={thStyle}>{t.compareRuns.field}</th>
            {details.map((d, i) => (
              <th key={d.run_id} style={{ ...thStyle, color: PALETTE[i % PALETTE.length] }}>
                {d.experiment_name}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {CONFIG_ROWS.map(({ id, pick }) => {
            const values = details.map((d) => pick(d.config));
            const reference = values[0];
            const anyDiff = values.some((v) => !sameConfigValue(v, reference));
            return (
              <tr key={id}>
                <td style={tdLabelStyle}>{t.compareRuns.config[id]}</td>
                {values.map((v, i) => {
                  const isDifferent = anyDiff && i > 0 && !sameConfigValue(v, reference);
                  return (
                    <td
                      key={details[i].run_id}
                      style={{
                        ...tdStyle,
                        ...(isDifferent
                          ? {
                              background: "oklch(0.78 0.16 75 / 0.16)",
                              color: "oklch(0.92 0.14 75)",
                              fontWeight: 600,
                            }
                          : {}),
                      }}
                    >
                      {fmtConfigValue(v)}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function sameConfigValue(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (a === null || a === undefined) return b === null || b === undefined;
  return String(a) === String(b);
}

/** When any run has a preprocessing pipeline, render each one as an ordered
 * list side by side so the differences are inspectable at a glance. */
function PreprocessingCompare({ details }: { details: RunDetail[] }) {
  const t = useT();
  const pipelines = details.map((d) => pickPreprocessingSteps(d.config));
  const anyPipeline = pipelines.some((p) => p.length > 0);
  if (!anyPipeline) return null;

  return (
    <div
      style={{
        padding: 14,
        background: "rgba(255,255,255,0.025)",
        border: "1px solid var(--vf-panel-stroke)",
        borderRadius: 12,
      }}
    >
      <div
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: 10,
          letterSpacing: "0.16em",
          textTransform: "uppercase",
          color: "var(--vf-text-muted)",
          marginBottom: 10,
        }}
      >
        {t.compareRuns.preprocessingTitle}
      </div>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: `repeat(${details.length}, minmax(0, 1fr))`,
          gap: 12,
        }}
      >
        {details.map((d, i) => {
          const steps = pipelines[i];
          const color = PALETTE[i % PALETTE.length];
          return (
            <div
              key={d.run_id}
              style={{
                padding: 10,
                background: "rgba(0,0,0,0.25)",
                border: "1px solid var(--vf-panel-stroke)",
                borderRadius: 10,
              }}
            >
              <div
                style={{
                  fontFamily: "var(--font-mono)",
                  fontSize: 10,
                  letterSpacing: "0.10em",
                  color,
                  marginBottom: 6,
                }}
              >
                {d.experiment_name}
              </div>
              {steps.length === 0 ? (
                <div
                  style={{
                    fontFamily: "var(--font-mono)",
                    fontSize: 11,
                    color: "var(--vf-text-muted)",
                    fontStyle: "italic",
                  }}
                >
                  {t.compareRuns.noPreprocessing}
                </div>
              ) : (
                <ol
                  style={{
                    margin: 0,
                    paddingLeft: 22,
                    display: "flex",
                    flexDirection: "column",
                    gap: 4,
                  }}
                >
                  {steps.map((s, idx) => {
                    const { kind, ...rest } = s as { kind?: unknown } & Record<string, unknown>;
                    const params = Object.entries(rest)
                      .map(([k, v]) => `${k}=${v}`)
                      .join(", ");
                    return (
                      <li
                        key={idx}
                        style={{
                          fontFamily: "var(--font-mono)",
                          fontSize: 11,
                          color: "var(--vf-text)",
                        }}
                      >
                        <strong>{String(kind ?? "?")}</strong>
                        {params && (
                          <span style={{ color: "var(--vf-text-muted)", marginLeft: 6 }}>
                            ({params})
                          </span>
                        )}
                      </li>
                    );
                  })}
                </ol>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

/**
 * The epoch curves of the compared runs: the task's own series (lib/compare-curves.ts),
 * a chart each, with a picker when the runs measured more than the ones drawn first.
 *
 * A run with no per-epoch history (a replicate group keeps the mean of its seeds, not
 * their epochs) has no line; the runs left out are named instead of being silently
 * absent from the chart.
 */
function EpochCurves({ details, task }: { details: RunDetail[]; task: string }) {
  const t = useT();
  // null until the researcher picks: then the task's initial series are drawn.
  const [picked, setPicked] = useState<string[] | null>(null);
  const available = curveSeries(
    task,
    details.map((d) => d.history),
  );
  const shown = selectedCurves(available, picked);
  const accent = accentForTask(task);
  const without = details.filter((d) => d.history.length === 0);

  if (available.length === 0 && without.length === 0) return null;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      {without.length > 0 && (
        <div
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: 11,
            lineHeight: 1.5,
            color: "var(--vf-text-muted)",
          }}
        >
          {t.compareRuns.noCurves(
            without.map((d) => d.experiment_name).join(", "),
            without.some((d) => d.group != null),
          )}
        </div>
      )}
      {available.length > 1 && (
        <div
          role="group"
          aria-label={t.compareRuns.curvePicker}
          style={{ display: "flex", alignItems: "center", flexWrap: "wrap", gap: 6 }}
        >
          <span
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: 9,
              letterSpacing: "0.16em",
              textTransform: "uppercase",
              color: "var(--vf-text-muted)",
              marginRight: 4,
            }}
          >
            {t.compareRuns.curvePicker}
          </span>
          {available.map((s) => {
            const on = shown.some((x) => x.key === s.key);
            return (
              <button
                key={s.key}
                type="button"
                aria-pressed={on}
                onClick={() =>
                  setPicked(
                    toggleCurve(
                      shown.map((x) => x.key),
                      s.key,
                    ),
                  )
                }
                style={{
                  padding: "4px 10px",
                  background: on ? "rgba(255,255,255,0.08)" : "rgba(255,255,255,0.025)",
                  border: `1px solid ${on ? accent : "var(--vf-panel-stroke)"}`,
                  borderRadius: 999,
                  color: on ? "var(--vf-text)" : "var(--vf-text-dim)",
                  fontFamily: "var(--font-mono)",
                  fontSize: 11,
                  cursor: "pointer",
                }}
              >
                {s.label ? t.compareRuns.curves[s.label] : s.key}
              </button>
            );
          })}
        </div>
      )}
      {shown.map((s) => (
        <OverlayChart key={s.key} details={details} series={s} accent={accent} />
      ))}
    </div>
  );
}

function OverlayChart({
  details,
  series: curve,
  accent,
}: {
  details: RunDetail[];
  series: CurveSeries;
  accent: string;
}) {
  const t = useT();
  const title = t.compareRuns.curveTitle(curve.label ? t.compareRuns.curves[curve.label] : curve.key);
  const width = 720;
  const height = 220;
  const padding = { top: 16, right: 16, bottom: 28, left: 44 };
  const innerW = width - padding.left - padding.right;
  const innerH = height - padding.top - padding.bottom;

  // Collect series + ranges.
  const series = details
    .map((d, i) => {
      const points = d.history.map((h) => ({
        x: h.epoch,
        y: numericMetric(h[curve.key]) ?? NaN,
      }));
      return { runId: d.run_id, label: d.experiment_name, color: PALETTE[i % PALETTE.length], points };
    })
    .filter((s) => s.points.length > 0);

  if (series.length === 0) {
    return null;
  }

  const allX = series.flatMap((s) => s.points.map((p) => p.x));
  const allY = series.flatMap((s) => s.points.map((p) => p.y)).filter((v) => Number.isFinite(v));
  // Runs stopped before they measured anything have no curve to draw.
  if (allY.length === 0) return null;
  const xMin = Math.min(...allX);
  const xMax = Math.max(...allX);
  const yMin = Math.min(...allY);
  const yMax = Math.max(...allY);
  const yRange = yMax - yMin || 1;

  const xScale = (x: number) => padding.left + ((x - xMin) / Math.max(xMax - xMin, 1)) * innerW;
  const yScale = (y: number) =>
    padding.top + innerH - ((y - yMin) / yRange) * innerH;

  return (
    <div
      style={{
        padding: 14,
        background: "rgba(255,255,255,0.025)",
        border: "1px solid var(--vf-panel-stroke)",
        borderRadius: 12,
      }}
    >
      <div
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: 10,
          letterSpacing: "0.16em",
          textTransform: "uppercase",
          color: "var(--vf-text-muted)",
          marginBottom: 8,
        }}
      >
        <span style={{ color: accent }}>//</span> {title}
        {curve.direction && (
          <span style={{ marginLeft: 10, letterSpacing: "0.06em", textTransform: "none" }}>
            · {t.compareRuns.direction[curve.direction]}
          </span>
        )}
      </div>
      <svg width={width} height={height} style={{ display: "block", maxWidth: "100%" }}>
        {/* Y-axis labels (min/max) */}
        <text x={padding.left - 6} y={padding.top + 4} fontSize="10" fill="var(--vf-text-muted)" textAnchor="end">
          {yMax.toFixed(3)}
        </text>
        <text
          x={padding.left - 6}
          y={padding.top + innerH}
          fontSize="10"
          fill="var(--vf-text-muted)"
          textAnchor="end"
        >
          {yMin.toFixed(3)}
        </text>
        {/* X-axis labels */}
        <text
          x={padding.left}
          y={height - 8}
          fontSize="10"
          fill="var(--vf-text-muted)"
        >
          {xMin}
        </text>
        <text
          x={width - padding.right}
          y={height - 8}
          fontSize="10"
          fill="var(--vf-text-muted)"
          textAnchor="end"
        >
          {xMax}
        </text>
        {/* Grid */}
        <line
          x1={padding.left}
          y1={padding.top + innerH}
          x2={width - padding.right}
          y2={padding.top + innerH}
          stroke="var(--vf-panel-stroke)"
          strokeWidth="1"
        />
        <line
          x1={padding.left}
          y1={padding.top}
          x2={padding.left}
          y2={padding.top + innerH}
          stroke="var(--vf-panel-stroke)"
          strokeWidth="1"
        />
        {/* Lines */}
        {series.map((s) => {
          const finite = s.points.filter((p) => Number.isFinite(p.y));
          const path = finite
            .map((p, i) => `${i === 0 ? "M" : "L"} ${xScale(p.x)} ${yScale(p.y)}`)
            .join(" ");
          return (
            <g key={s.runId}>
              <path
                d={path}
                fill="none"
                stroke={s.color}
                strokeWidth="2"
                strokeLinejoin="round"
                strokeLinecap="round"
                opacity={0.92}
              />
              {/* One epoch (PatchCore fits in one) is a point, which a line does not draw. */}
              {finite.length === 1 && (
                <circle cx={xScale(finite[0].x)} cy={yScale(finite[0].y)} r={3.5} fill={s.color} />
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
}
