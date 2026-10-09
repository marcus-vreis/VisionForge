import { useState } from "react";
import { artifactUrl, downloadRunMarkdown } from "../api/client";
import { useT } from "../i18n/useT";
import { metricCi } from "../lib/metric-ci";
import { isCrossValidationReport } from "../lib/report-shape";
import { plannedUnits, unitPlan } from "../lib/unit-plan";
import { STOPPED_COLOR, countUnits, unitState } from "../lib/unit-status";
import type { MetricCI, RunResult } from "../types/run";
import { Lightbox } from "./Lightbox";
import { Rich } from "./Rich";

interface ResultsViewProps {
  result: RunResult;
  onClose: () => void;
  taskAccent: string;
  /** How many folds / models / trials / replicates the run was submitted with.
   *  A stopped job lists only the units that ran, and its header counts against
   *  this when the report does not say how many were planned. */
  submittedUnits?: number | null;
}

/** Format metric values for display. */
function formatMetric(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") {
    // A metric that was never measured arrives as null, but a NaN or an
    // infinity must not reach the screen either.
    if (!Number.isFinite(value)) return "—";
    return value % 1 === 0 ? String(value) : value.toFixed(4);
  }
  return String(value);
}

interface MetricCardProps {
  label: string;
  value: string;
  accent: string;
  highlight?: boolean;
  /** Bootstrap interval for this metric, when the run has one (ADR-074). */
  ci?: MetricCI;
}

/** `0.7294 – 0.7713` under the value, with the split size it was resampled from. */
function CiFootnote({ ci }: { ci: MetricCI }) {
  const t = useT();
  return (
    <div
      title={t.metrics.ciTooltip(
        Math.round(ci.confidence * 100),
        ci.n_resamples,
        ci.n_samples,
      )}
      style={{
        marginTop: 4,
        fontFamily: "var(--font-mono)",
        fontSize: 10,
        color: "var(--vf-text-muted)",
        whiteSpace: "nowrap",
        cursor: "help",
      }}
    >
      {ci.ci_low.toFixed(4)} – {ci.ci_high.toFixed(4)}
    </div>
  );
}

function MetricCard({ label, value, accent, highlight, ci }: MetricCardProps) {
  return (
    <div
      style={{
        padding: 14,
        borderRadius: 12,
        border: "1px solid var(--vf-panel-stroke)",
        background: highlight
          ? `linear-gradient(180deg, ${accent}22 0%, rgba(12,14,18,0.5) 100%)`
          : "rgba(12,14,18,0.55)",
        position: "relative",
        overflow: "hidden",
      }}
    >
      {highlight && (
        <span
          style={{
            position: "absolute",
            top: 10,
            right: 10,
            width: 6,
            height: 6,
            borderRadius: "50%",
            background: accent,
            boxShadow: `0 0 10px ${accent}`,
          }}
        />
      )}
      <div
        style={{
          fontSize: 10,
          letterSpacing: "0.18em",
          textTransform: "uppercase",
          color: "var(--vf-text-muted)",
          fontFamily: "var(--font-mono)",
        }}
      >
        {label}
      </div>
      <div
        style={{
          fontSize: 22,
          marginTop: 6,
          fontFamily: "var(--font-mono)",
          fontWeight: 600,
          color: highlight ? accent : "var(--vf-text)",
        }}
      >
        {value}
      </div>
      {ci && <CiFootnote ci={ci} />}
    </div>
  );
}

/** Results sheet that slides up over the param panel. */
export function ResultsView({
  result,
  onClose,
  taskAccent,
  submittedUnits = null,
}: ResultsViewProps) {
  const t = useT();
  const metricLabels: Record<string, string> = t.resultsView.metricLabels;
  const plotLabels: Record<string, string> = t.plots.labels;
  const graphics = result.artifacts?.graphics ?? [];
  const metricsEntries = Object.entries(result.metrics);
  const [lightbox, setLightbox] = useState<{ src: string; caption: string } | null>(null);

  // Pick the "best" metric for highlighting — prefer accuracy, then f1, then first
  const highlightKey =
    metricsEntries.find(([k]) => k === "test_accuracy")?.[0] ??
    metricsEntries.find(([k]) => k === "test_f1")?.[0] ??
    metricsEntries[0]?.[0];

  return (
    <div
      style={{
        position: "relative",
        animation: "sheetIn 360ms cubic-bezier(0.2, 0.9, 0.2, 1) forwards",
        padding: 28,
        background: "rgba(10,12,16,0.55)",
        backdropFilter: "blur(20px) saturate(140%)",
        WebkitBackdropFilter: "blur(20px) saturate(140%)",
        border: "1px solid var(--vf-panel-stroke)",
        borderRadius: 20,
        boxShadow:
          "0 30px 80px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.04)",
      }}
    >
      {/* Header */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 16,
          marginBottom: 22,
        }}
      >
        <div style={{ flex: 1 }}>
          <div
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: 10,
              letterSpacing: "0.22em",
              textTransform: "uppercase",
              color: "var(--vf-text-muted)",
              marginBottom: 6,
            }}
          >
            {t.resultsView.title}
          </div>
          <div
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: 18,
              fontWeight: 600,
              color: "var(--vf-text)",
            }}
          >
            {result.run_id}
          </div>
        </div>
        <button
          type="button"
          onClick={() => void downloadRunMarkdown(result.run_id)}
          title={t.modelCard.title}
          style={{
            padding: "8px 14px",
            background: "var(--accent-soft)",
            border: `1px solid ${taskAccent}`,
            borderRadius: 10,
            color: "var(--vf-text)",
            fontFamily: "var(--font-mono)",
            fontSize: 11,
            letterSpacing: "0.10em",
            textTransform: "uppercase",
            cursor: "pointer",
          }}
        >
          {t.modelCard.button}
        </button>
        <button
          type="button"
          onClick={onClose}
          style={{
            width: 36,
            height: 36,
            borderRadius: "50%",
            border: "1px solid var(--vf-panel-stroke)",
            background: "rgba(255,255,255,0.03)",
            color: "var(--vf-text)",
            fontSize: 20,
            lineHeight: "1",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            cursor: "pointer",
          }}
        >
          ×
        </button>
      </div>

      {/* Metrics grid */}
      {metricsEntries.length > 0 && (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))",
            gap: 12,
            marginBottom: 22,
          }}
        >
          {metricsEntries.map(([key, value]) => (
            <MetricCard
              key={key}
              label={metricLabels[key] ?? key}
              value={formatMetric(value)}
              accent={taskAccent}
              highlight={key === highlightKey}
              ci={metricCi(result.metric_cis, key)}
            />
          ))}
        </div>
      )}

      {/* Plot images */}
      {graphics.length > 0 && (
        <div>
          <div
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: 10,
              letterSpacing: "0.20em",
              textTransform: "uppercase",
              color: "var(--vf-text-muted)",
              marginBottom: 14,
            }}
          >
            {t.resultsView.graphsTitle}
          </div>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))",
              gap: 16,
            }}
          >
            {graphics.map((path, idx) => {
              const filename = path.replace(/\\/g, "/").split("/").pop() ?? path;
              const label = plotLabels[filename] ?? filename;
              const url = artifactUrl(path);
              return (
                <button
                  key={idx}
                  type="button"
                  onClick={() => setLightbox({ src: url, caption: label })}
                  style={{
                    borderRadius: 12,
                    border: `1px solid ${taskAccent}33`,
                    overflow: "hidden",
                    background: "rgba(0,0,0,0.3)",
                    padding: 0,
                    cursor: "zoom-in",
                    display: "flex",
                    flexDirection: "column",
                    textAlign: "left",
                  }}
                >
                  <img
                    src={url}
                    alt={label}
                    style={{ width: "100%", height: "auto", display: "block" }}
                  />
                  <div
                    style={{
                      padding: "8px 12px 10px",
                      fontFamily: "var(--font-mono)",
                      fontSize: 11,
                      color: "var(--vf-text-dim)",
                    }}
                  >
                    {label}
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      )}

      {lightbox && (
        <Lightbox
          src={lightbox.src}
          caption={lightbox.caption}
          onClose={() => setLightbox(null)}
        />
      )}

      {/* Report summary — branches between CV/comparison/grid (structured) and generic JSON */}
      {result.report && Object.keys(result.report).length > 0 && (
        isCrossValidationReport(result.report) ? (
          <CrossValidationReport
            report={result.report}
            accent={taskAccent}
            submitted={submittedUnits}
          />
        ) : isTaskCvReport(result.report) ? (
          <TaskCvReport
            report={result.report}
            accent={taskAccent}
            submitted={submittedUnits}
          />
        ) : isReplicatesReport(result.report) ? (
          <ReplicatesReport
            report={result.report}
            accent={taskAccent}
            submitted={submittedUnits}
          />
        ) : isTaskComparisonReport(result.report) ? (
          <TaskComparisonReport
            report={result.report}
            accent={taskAccent}
            submitted={submittedUnits}
          />
        ) : isTaskSweepReport(result.report) ? (
          <TaskSweepReport
            report={result.report}
            accent={taskAccent}
            submitted={submittedUnits}
          />
        ) : isModelComparisonReport(result.report) ? (
          <ModelComparisonReport
            report={result.report}
            accent={taskAccent}
            submitted={submittedUnits}
          />
        ) : isGridSearchReport(result.report) ? (
          <GridSearchReport
            report={result.report}
            accent={taskAccent}
            submitted={submittedUnits}
          />
        ) : (
          <div style={{ marginTop: 22 }}>
            <div
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: 10,
                letterSpacing: "0.20em",
                textTransform: "uppercase",
                color: "var(--vf-text-muted)",
                marginBottom: 14,
              }}
            >
              {t.resultsView.reportTitle}
            </div>
            <pre
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: 11,
                background: "rgba(0,0,0,0.45)",
                border: "1px solid var(--vf-panel-stroke)",
                borderRadius: 10,
                padding: "12px 14px",
                overflowX: "auto",
                color: "var(--vf-text-dim)",
                lineHeight: 1.6,
              }}
            >
              {JSON.stringify(result.report, null, 2)}
            </pre>
          </div>
        )
      )}
    </div>
  );
}

interface FoldRecord {
  fold: number;
  train_size: number;
  val_size: number;
  status: string;
  error: string;
  best_val_loss: number | null;
  accuracy: number | null;
  f1: number | null;
}

/** Structured render for CrossValidationBlock.report().
 *
 * Beats the JSON dump on three axes: highlights mean ± std (the headline number
 * in any K-Fold paper), shows per-fold accuracy/f1 in a table, and surfaces
 * failed folds explicitly so a partial-success run isn't silently averaged
 * away. */
function CrossValidationReport({
  report,
  accent,
  submitted = null,
}: {
  report: Record<string, unknown>;
  accent: string;
  submitted?: number | null;
}) {
  const t = useT();
  const folds = (report["fold_results"] as FoldRecord[]) ?? [];
  // Null when too few folds finished: no mean before one, no spread before two.
  const meanAcc = report["mean_accuracy"] as number | null;
  const stdAcc = report["std_accuracy"] as number | null;
  const meanF1 = report["mean_f1"] as number | null;
  const stdF1 = report["std_f1"] as number | null;

  // Against the folds planned, not only those that ran: a K-fold stopped in its
  // first fold lists one.
  const plan = unitPlan(folds.length, plannedUnits(report, submitted));
  const successful = folds.filter((f) => unitState(f.status) === "ok");
  const failed = countUnits(folds, "failed");
  const stopped = countUnits(folds, "stopped");

  return (
    <div style={{ marginTop: 22, display: "flex", flexDirection: "column", gap: 18 }}>
      <div
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: 10,
          letterSpacing: "0.20em",
          textTransform: "uppercase",
          color: "var(--vf-text-muted)",
        }}
      >
        {t.resultsView.cv.title(successful.length, plan.total, failed, stopped)}
        {plan.notRun > 0 && t.resultsView.notRun(plan.notRun)}
      </div>

      {/* Headline: mean ± std for accuracy and F1 */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
          gap: 12,
        }}
      >
        <AggregateCard
          label={t.resultsView.cv.accuracyMeanStd}
          mean={meanAcc}
          std={stdAcc}
          accent={accent}
          highlight
        />
        <AggregateCard
          label={t.resultsView.cv.f1MeanStd}
          mean={meanF1}
          std={stdF1}
          accent={accent}
        />
      </div>

      {/* Per-fold table */}
      <div
        style={{
          padding: 14,
          background: "rgba(255,255,255,0.025)",
          border: "1px solid var(--vf-panel-stroke)",
          borderRadius: 12,
          overflowX: "auto",
        }}
      >
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
              <th style={cvThStyle}>{t.resultsView.cv.fold}</th>
              <th style={cvThStyle}>{t.resultsView.cv.trainSize}</th>
              <th style={cvThStyle}>{t.resultsView.cv.valSize}</th>
              <th style={cvThStyle}>{t.resultsView.cv.valLoss}</th>
              <th style={cvThStyle}>{t.resultsView.cv.accuracy}</th>
              <th style={cvThStyle}>F1</th>
              <th style={cvThStyle}>{t.resultsView.cols.status}</th>
            </tr>
          </thead>
          <tbody>
            {folds.map((f) => {
              const state = unitState(f.status);
              return (
                <tr key={f.fold}>
                  <td style={cvTdLabelStyle}>#{f.fold + 1}</td>
                  <td style={cvTdStyle}>{f.train_size}</td>
                  <td style={cvTdStyle}>{f.val_size}</td>
                  <td style={cvTdStyle}>{formatMetric(f.best_val_loss)}</td>
                  <td style={cvTdStyle}>{formatMetric(f.accuracy)}</td>
                  <td style={cvTdStyle}>{formatMetric(f.f1)}</td>
                  <td
                    style={{
                      ...cvTdStyle,
                      color:
                        state === "ok"
                          ? "oklch(0.85 0.16 150)"
                          : state === "stopped"
                            ? STOPPED_COLOR
                            : "oklch(0.85 0.14 22)",
                    }}
                  >
                    {state === "ok"
                      ? "✓"
                      : state === "stopped"
                        ? `■ ${t.resultsView.outcome.stopped}`
                        : `× ${f.error || "?"}`}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function AggregateCard({
  label,
  mean,
  std,
  accent,
  highlight,
}: {
  label: string;
  mean: number | null;
  std: number | null;
  accent: string;
  highlight?: boolean;
}) {
  return (
    <div
      style={{
        padding: 16,
        borderRadius: 12,
        border: "1px solid var(--vf-panel-stroke)",
        background: highlight
          ? `linear-gradient(180deg, ${accent}22 0%, rgba(12,14,18,0.5) 100%)`
          : "rgba(12,14,18,0.55)",
      }}
    >
      <div
        style={{
          fontSize: 10,
          letterSpacing: "0.18em",
          textTransform: "uppercase",
          color: "var(--vf-text-muted)",
          fontFamily: "var(--font-mono)",
        }}
      >
        {label}
      </div>
      <div
        style={{
          fontSize: 24,
          marginTop: 6,
          fontFamily: "var(--font-mono)",
          fontWeight: 600,
          color: highlight ? accent : "var(--vf-text)",
        }}
      >
        {formatMetric(mean)}
        <span
          style={{
            fontSize: 14,
            color: "var(--vf-text-muted)",
            marginLeft: 6,
            fontWeight: 400,
          }}
        >
          ± {formatMetric(std)}
        </span>
      </div>
    </div>
  );
}

interface TaskCvFoldRow {
  fold: number;
  status: string;
  train_size: number;
  val_size: number;
  metrics: Record<string, number>;
  error: string;
}

/** Standalone-task K-fold report (ADR-050): `fold_results` + per-metric
 *  `aggregate` (the classification CV report carries `mean_accuracy` instead). */
function isTaskCvReport(report: Record<string, unknown>): boolean {
  return (
    Array.isArray(report["fold_results"]) &&
    typeof report["aggregate"] === "object" &&
    report["aggregate"] !== null
  );
}

/** Fold-a-fold table + mean ± std headline for a standalone-task K-fold run. */
function TaskCvReport({
  report,
  accent,
  submitted = null,
}: {
  report: Record<string, unknown>;
  accent: string;
  submitted?: number | null;
}) {
  const t = useT();
  const folds = (report["fold_results"] as TaskCvFoldRow[]) ?? [];
  const aggregate =
    (report["aggregate"] as Record<
      string,
      { mean: number | null; std: number | null; n?: number }
    >) ?? {};
  const metric = report["metric"] as string;
  const nFolds = report["n_folds"] as number;
  const ok = report["successful_folds"] as number;
  const stoppedFolds = countUnits(folds, "stopped");
  const foldPlan = unitPlan(folds.length, plannedUnits(report, submitted ?? nFolds));
  const headline = aggregate[metric];
  const metricKeys = Object.keys(aggregate);

  return (
    <div style={{ marginTop: 22, display: "flex", flexDirection: "column", gap: 16 }}>
      <div
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: 10,
          letterSpacing: "0.20em",
          textTransform: "uppercase",
          color: "var(--vf-text-muted)",
        }}
      >
        {t.resultsView.taskCv.title(ok, foldPlan.total, metric, stoppedFolds)}
        {foldPlan.notRun > 0 && t.resultsView.notRun(foldPlan.notRun)}
      </div>

      {headline && (
        <div
          style={{
            padding: 16,
            background: `linear-gradient(180deg, ${accent}1c 0%, rgba(12,14,18,0.5) 100%)`,
            border: `1px solid ${accent}55`,
            borderRadius: 12,
            fontFamily: "var(--font-mono)",
            fontSize: 20,
            color: "var(--vf-text)",
          }}
        >
          {metric} = <span style={{ color: accent }}>{formatMetric(headline.mean)}</span>
          <span style={{ color: "var(--vf-text-dim)" }}> ± {formatMetric(headline.std)}</span>
          <span style={{ fontSize: 11, color: "var(--vf-text-muted)", marginLeft: 10 }}>
            {t.resultsView.taskCv.meanStd}
            {typeof headline.n === "number" && ` · ${t.resultsView.taskCv.sample(headline.n)}`}
          </span>
        </div>
      )}

      <div
        style={{
          padding: 14,
          background: "rgba(255,255,255,0.025)",
          border: "1px solid var(--vf-panel-stroke)",
          borderRadius: 12,
          overflowX: "auto",
        }}
      >
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
              <th style={cvThStyle}>{t.resultsView.taskCv.fold}</th>
              {metricKeys.map((k) => (
                <th key={k} style={cvThStyle}>
                  {k}
                </th>
              ))}
              <th style={cvThStyle}>{t.resultsView.taskCv.trainVal}</th>
              <th style={cvThStyle}>{t.resultsView.cols.status}</th>
            </tr>
          </thead>
          <tbody>
            {folds.map((f) => (
              <tr key={f.fold}>
                <td style={cvTdLabelStyle}>{f.fold + 1}</td>
                {metricKeys.map((k) => (
                  <td key={k} style={cvTdStyle}>
                    {formatMetric(f.metrics?.[k])}
                  </td>
                ))}
                <td style={cvTdStyle}>
                  {f.train_size}/{f.val_size}
                </td>
                <td
                  style={{
                    ...cvTdStyle,
                    color: unitColor(unitState(f.status)),
                  }}
                >
                  {unitText(t, unitState(f.status), f.error)}
                </td>
              </tr>
            ))}
          </tbody>
          <tfoot>
            <tr>
              <td style={{ ...cvTdLabelStyle, color: accent }}>μ ± σ</td>
              {metricKeys.map((k) => (
                <td key={k} style={{ ...cvTdStyle, color: "var(--vf-text)" }}>
                  {formatMetric(aggregate[k].mean)} ± {formatMetric(aggregate[k].std)}
                </td>
              ))}
              <td style={cvTdStyle} />
              <td style={cvTdStyle} />
            </tr>
          </tfoot>
        </table>
      </div>
    </div>
  );
}

interface ReplicateTrialRow {
  seed: number;
  status: string;
  metrics: Record<string, number>;
  training_time_s: number | null;
  error: string;
}

interface ReplicateAggregate {
  n: number;
  mean: number;
  std: number | null;
  min: number;
  max: number;
  ci95_low: number | null;
  ci95_high: number | null;
}

/** Multi-seed replicates report (ADR-056): identified by the `seeds` array +
 *  per-metric `aggregates` — must be tested before the comparison/sweep shapes
 *  (it also carries `metric` + `trials`). */
function isReplicatesReport(report: Record<string, unknown>): boolean {
  return (
    Array.isArray(report["seeds"]) &&
    typeof report["aggregates"] === "object" &&
    report["aggregates"] !== null
  );
}

/** Headline mean ± CI + per-metric aggregates + per-seed table. */
function ReplicatesReport({
  report,
  accent,
  submitted = null,
}: {
  report: Record<string, unknown>;
  accent: string;
  submitted?: number | null;
}) {
  const t = useT();
  const trials = (report["trials"] as ReplicateTrialRow[]) ?? [];
  const metric = report["metric"] as string;
  const aggregates =
    (report["aggregates"] as Record<string, ReplicateAggregate>) ?? {};
  const headline = report["headline"] as ReplicateAggregate | null;
  const total = report["total_replicates"] as number;
  const ok = report["successful_replicates"] as number;
  const stoppedReplicates = countUnits(trials, "stopped");
  const plan = unitPlan(total, plannedUnits(report, submitted));

  const ciHalf =
    headline && headline.ci95_high !== null && headline.ci95_low !== null
      ? (headline.ci95_high - headline.ci95_low) / 2
      : null;

  return (
    <div style={{ marginTop: 22, display: "flex", flexDirection: "column", gap: 16 }}>
      <div
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: 10,
          letterSpacing: "0.20em",
          textTransform: "uppercase",
          color: "var(--vf-text-muted)",
        }}
      >
        {t.resultsView.replicates.title(ok, plan.total, metric, stoppedReplicates)}
        {plan.notRun > 0 && t.resultsView.notRun(plan.notRun)}
      </div>

      {headline && (
        <div
          style={{
            padding: 16,
            background: `linear-gradient(180deg, ${accent}1c 0%, rgba(12,14,18,0.5) 100%)`,
            border: `1px solid ${accent}55`,
            borderRadius: 12,
            fontFamily: "var(--font-mono)",
          }}
        >
          <div
            style={{
              fontSize: 9,
              letterSpacing: "0.14em",
              textTransform: "uppercase",
              color: "var(--vf-text-muted)",
              marginBottom: 8,
            }}
          >
            {t.resultsView.replicates.citable}
          </div>
          <div style={{ fontSize: 20, color: "var(--vf-text)" }}>
            {metric} = <span style={{ color: accent }}>{formatMetric(headline.mean)}</span>
            {ciHalf !== null && (
              <span style={{ color: "var(--vf-text-dim)" }}> ± {formatMetric(ciHalf)}</span>
            )}
            <span style={{ fontSize: 11, color: "var(--vf-text-muted)", marginLeft: 10 }}>
              {t.resultsView.replicates.headlineMeta(ciHalf !== null, headline.n)}
            </span>
          </div>
        </div>
      )}

      <div
        style={{
          padding: 14,
          background: "rgba(255,255,255,0.025)",
          border: "1px solid var(--vf-panel-stroke)",
          borderRadius: 12,
          overflowX: "auto",
        }}
      >
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
              <th style={cvThStyle}>{t.resultsView.replicates.metric}</th>
              <th style={cvThStyle}>{t.resultsView.replicates.n}</th>
              <th style={cvThStyle}>{t.resultsView.replicates.mean}</th>
              <th style={cvThStyle}>{t.resultsView.replicates.std}</th>
              <th style={cvThStyle}>{t.resultsView.replicates.min}</th>
              <th style={cvThStyle}>{t.resultsView.replicates.max}</th>
              <th style={cvThStyle}>{t.resultsView.replicates.ci}</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(aggregates).map(([key, agg]) => (
              <tr key={key}>
                <td
                  style={{
                    ...cvTdLabelStyle,
                    color: key === metric ? accent : "var(--vf-text-muted)",
                    fontWeight: key === metric ? 700 : 500,
                  }}
                >
                  {key}
                </td>
                <td style={cvTdStyle}>{agg.n}</td>
                <td style={cvTdStyle}>{formatMetric(agg.mean)}</td>
                <td style={cvTdStyle}>{agg.std === null ? "—" : formatMetric(agg.std)}</td>
                <td style={cvTdStyle}>{formatMetric(agg.min)}</td>
                <td style={cvTdStyle}>{formatMetric(agg.max)}</td>
                <td style={cvTdStyle}>
                  {agg.ci95_low === null || agg.ci95_high === null
                    ? "—"
                    : `[${formatMetric(agg.ci95_low)}, ${formatMetric(agg.ci95_high)}]`}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div
        style={{
          padding: 14,
          background: "rgba(255,255,255,0.025)",
          border: "1px solid var(--vf-panel-stroke)",
          borderRadius: 12,
          overflowX: "auto",
        }}
      >
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
              <th style={cvThStyle}>{t.resultsView.cols.seed}</th>
              <th style={cvThStyle}>{metric}</th>
              <th style={cvThStyle}>{t.resultsView.cols.time}</th>
              <th style={cvThStyle}>{t.resultsView.cols.status}</th>
            </tr>
          </thead>
          <tbody>
            {trials.map((trial) => (
              <tr key={trial.seed}>
                <td style={cvTdLabelStyle}>{trial.seed}</td>
                <td style={cvTdStyle}>{formatMetric(trial.metrics?.[metric])}</td>
                <td style={cvTdStyle}>
                  {typeof trial.training_time_s === "number"
                    ? trial.training_time_s.toFixed(1)
                    : "—"}
                </td>
                <td
                  style={{
                    ...cvTdStyle,
                    color: unitColor(unitState(trial.status)),
                  }}
                >
                  {unitText(t, unitState(trial.status), trial.error)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

interface TaskComparisonTrial {
  model_arch: string;
  status: string;
  metrics: Record<string, number>;
  training_time_s: number | null;
  error: string;
}

/** Standalone-task comparison report (ADR-044): has a string `metric` and a
 *  `trials` array with nested per-task metrics — distinct from the classification
 *  ModelComparisonReport (flat accuracy/f1/auc_roc, no `metric` key). */
function isTaskComparisonReport(report: Record<string, unknown>): boolean {
  return (
    typeof report["metric"] === "string" &&
    Array.isArray(report["trials"]) &&
    report["mode"] === undefined
  );
}

/** Ranked architecture table for a regression/segmentation model comparison. */
function TaskComparisonReport({
  report,
  accent,
  submitted = null,
}: {
  report: Record<string, unknown>;
  accent: string;
  submitted?: number | null;
}) {
  const t = useT();
  const trials = (report["trials"] as TaskComparisonTrial[]) ?? [];
  const metric = report["metric"] as string;
  const totalRan = report["total_ran"] as number;
  const failedCount = countUnits(trials, "failed");
  const stoppedCount = countUnits(trials, "stopped");
  const plan = unitPlan(totalRan, plannedUnits(report, submitted));

  const successful = trials.filter((trial) => unitState(trial.status) === "ok");
  const otherKeys = Array.from(
    new Set(successful.flatMap((trial) => Object.keys(trial.metrics ?? {}))),
  ).filter((k) => k !== metric);
  const metricCols = [metric, ...otherKeys];

  return (
    <div style={{ marginTop: 22, display: "flex", flexDirection: "column", gap: 18 }}>
      <div
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: 10,
          letterSpacing: "0.20em",
          textTransform: "uppercase",
          color: "var(--vf-text-muted)",
        }}
      >
        {t.resultsView.comparison.title(
          successful.length,
          plan.total,
          failedCount,
          stoppedCount,
          metric,
        )}
        {plan.notRun > 0 && t.resultsView.notRun(plan.notRun)}
      </div>

      <div
        style={{
          padding: 14,
          background: "rgba(255,255,255,0.025)",
          border: "1px solid var(--vf-panel-stroke)",
          borderRadius: 12,
          overflowX: "auto",
        }}
      >
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
              <th style={cvThStyle}>{t.resultsView.cols.rank}</th>
              <th style={cvThStyle}>{t.resultsView.cols.architecture}</th>
              {metricCols.map((k) => (
                <th key={k} style={cvThStyle}>
                  {k}
                </th>
              ))}
              <th style={cvThStyle}>{t.resultsView.cols.time}</th>
              <th style={cvThStyle}>{t.resultsView.cols.status}</th>
            </tr>
          </thead>
          <tbody>
            {trials.map((trial, i) => {
              const state = unitState(trial.status);
              const ok = state === "ok";
              return (
                <tr key={trial.model_arch}>
                  <td
                    style={{
                      ...cvTdLabelStyle,
                      color: i === 0 && ok ? accent : "var(--vf-text-muted)",
                      fontWeight: i === 0 && ok ? 700 : 500,
                    }}
                  >
                    {ok ? `#${i + 1}` : "—"}
                    {i === 0 && ok && <span style={{ marginLeft: 6 }}>👑</span>}
                  </td>
                  <td
                    style={{
                      ...cvTdStyle,
                      color: i === 0 && ok ? accent : "var(--vf-text)",
                      fontWeight: i === 0 && ok ? 600 : 400,
                    }}
                  >
                    {trial.model_arch}
                  </td>
                  {metricCols.map((k) => (
                    <td key={k} style={cvTdStyle}>
                      {formatMetric(trial.metrics?.[k])}
                    </td>
                  ))}
                  <td style={cvTdStyle}>
                    {typeof trial.training_time_s === "number"
                      ? trial.training_time_s.toFixed(1)
                      : "—"}
                  </td>
                  <td
                    style={{
                      ...cvTdStyle,
                      color: markColor(state),
                    }}
                  >
                    {markText(t, state, trial.error)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

interface TaskSweepTrial {
  trial_index: number;
  overrides: Record<string, unknown>;
  status: string;
  metrics: Record<string, number>;
  training_time_s: number | null;
  error: string;
}

/** Standalone-task sweep report (ADR-045): identified by a string `mode`
 *  (grid/random) plus a `trials` array. */
function isTaskSweepReport(report: Record<string, unknown>): boolean {
  return typeof report["mode"] === "string" && Array.isArray(report["trials"]);
}

function OverrideChips({ overrides }: { overrides: Record<string, unknown> }) {
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
      {Object.entries(overrides).map(([k, v]) => (
        <span
          key={k}
          style={{
            padding: "3px 8px",
            background: "rgba(0,0,0,0.30)",
            border: "1px solid var(--vf-panel-stroke)",
            borderRadius: 8,
            fontFamily: "var(--font-mono)",
            fontSize: 10.5,
            color: "var(--vf-text)",
          }}
        >
          <span style={{ color: "var(--vf-text-muted)" }}>{k.split(".").at(-1)}=</span>
          {String(v)}
        </span>
      ))}
    </div>
  );
}

/** Best trial + ranked table for a regression/segmentation hyperparameter sweep. */
function TaskSweepReport({
  report,
  accent,
  submitted = null,
}: {
  report: Record<string, unknown>;
  accent: string;
  submitted?: number | null;
}) {
  const t = useT();
  const trials = (report["trials"] as TaskSweepTrial[]) ?? [];
  const mode = report["mode"] as string;
  const metric = report["metric"] as string;
  const total = report["total_trials"] as number;
  const successful = report["successful_trials"] as number;
  const stoppedTrials = countUnits(trials, "stopped");
  const plan = unitPlan(total, plannedUnits(report, submitted));
  const best = report["best_trial"] as TaskSweepTrial | null;

  return (
    <div style={{ marginTop: 22, display: "flex", flexDirection: "column", gap: 16 }}>
      <div
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: 10,
          letterSpacing: "0.20em",
          textTransform: "uppercase",
          color: "var(--vf-text-muted)",
        }}
      >
        {t.resultsView.sweep.title(mode, successful, plan.total, stoppedTrials, metric)}
        {plan.notRun > 0 && t.resultsView.notRun(plan.notRun)}
      </div>

      {best && (
        <div
          style={{
            padding: 16,
            background: `linear-gradient(180deg, ${accent}1c 0%, rgba(12,14,18,0.5) 100%)`,
            border: `1px solid ${accent}55`,
            borderRadius: 12,
            display: "flex",
            flexDirection: "column",
            gap: 12,
          }}
        >
          <div
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: 9,
              letterSpacing: "0.14em",
              textTransform: "uppercase",
              color: "var(--vf-text-muted)",
            }}
          >
            {t.resultsView.sweep.best(metric)}
            <span style={{ color: accent, marginLeft: 4 }}>
              {formatMetric(best.metrics?.[metric])}
            </span>
          </div>
          <OverrideChips overrides={best.overrides} />
        </div>
      )}

      <div
        style={{
          padding: 14,
          background: "rgba(255,255,255,0.025)",
          border: "1px solid var(--vf-panel-stroke)",
          borderRadius: 12,
          overflowX: "auto",
        }}
      >
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
              <th style={cvThStyle}>{t.resultsView.cols.rank}</th>
              <th style={cvThStyle}>{metric}</th>
              <th style={cvThStyle}>{t.resultsView.cols.overrides}</th>
              <th style={cvThStyle}>{t.resultsView.cols.time}</th>
              <th style={cvThStyle}>{t.resultsView.cols.status}</th>
            </tr>
          </thead>
          <tbody>
            {trials.map((trial, i) => {
              const state = unitState(trial.status);
              const ok = state === "ok";
              return (
                <tr key={trial.trial_index}>
                  <td
                    style={{
                      ...cvTdLabelStyle,
                      color: i === 0 && ok ? accent : "var(--vf-text-muted)",
                      fontWeight: i === 0 && ok ? 700 : 500,
                    }}
                  >
                    {ok ? `#${i + 1}` : "—"}
                  </td>
                  <td style={{ ...cvTdStyle, color: i === 0 && ok ? accent : "var(--vf-text)" }}>
                    {formatMetric(trial.metrics?.[metric])}
                  </td>
                  <td style={cvTdStyle}>
                    <OverrideChips overrides={trial.overrides} />
                  </td>
                  <td style={cvTdStyle}>
                    {typeof trial.training_time_s === "number"
                      ? trial.training_time_s.toFixed(1)
                      : "—"}
                  </td>
                  <td
                    style={{
                      ...cvTdStyle,
                      color: markColor(state),
                    }}
                  >
                    {markText(t, state, trial.error)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

interface ModelComparisonTrial {
  model_arch: string;
  status: string;
  error?: string;
  accuracy: number | null;
  f1: number | null;
  auc_roc: number | null;
  training_time_s: number | null;
}

function isModelComparisonReport(report: Record<string, unknown>): boolean {
  return (
    Array.isArray(report["top_3"]) &&
    typeof report["total_ran"] === "number" &&
    typeof report["failed_count"] === "number"
  );
}

/** Structured render for ModelComparisonBlock.report().
 *
 * Surfaces the ranked top-3 architectures plus run totals. Each row shows
 * the trial's metric, training time, and status — failed rows stand out so
 * a partial-success comparison isn't mistaken for a clean sweep.
 */
function ModelComparisonReport({
  report,
  accent,
  submitted = null,
}: {
  report: Record<string, unknown>;
  accent: string;
  submitted?: number | null;
}) {
  const t = useT();
  const top3 = (report["top_3"] as ModelComparisonTrial[]) ?? [];
  const totalRan = report["total_ran"] as number;
  const failedCount = report["failed_count"] as number;
  // Absent from a report written before a stop could cut a model (ADR-111).
  const stoppedCount =
    typeof report["stopped_count"] === "number" ? report["stopped_count"] : 0;
  const plan = unitPlan(totalRan, plannedUnits(report, submitted));

  return (
    <div style={{ marginTop: 22, display: "flex", flexDirection: "column", gap: 18 }}>
      <div
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: 10,
          letterSpacing: "0.20em",
          textTransform: "uppercase",
          color: "var(--vf-text-muted)",
        }}
      >
        {t.resultsView.modelComparison.title(
          totalRan - failedCount - stoppedCount,
          plan.total,
          failedCount,
          stoppedCount,
        )}
        {plan.notRun > 0 && t.resultsView.notRun(plan.notRun)}
      </div>

      <div
        style={{
          padding: 14,
          background: "rgba(255,255,255,0.025)",
          border: "1px solid var(--vf-panel-stroke)",
          borderRadius: 12,
          overflowX: "auto",
        }}
      >
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
              <th style={cvThStyle}>{t.resultsView.cols.rank}</th>
              <th style={cvThStyle}>{t.resultsView.cols.architecture}</th>
              <th style={cvThStyle}>{t.resultsView.modelComparison.accuracy}</th>
              <th style={cvThStyle}>F1</th>
              <th style={cvThStyle}>{t.resultsView.modelComparison.aucRoc}</th>
              <th style={cvThStyle}>{t.resultsView.cols.time}</th>
            </tr>
          </thead>
          <tbody>
            {top3.map((trial, i) => {
              const isFirst = i === 0;
              return (
                <tr key={trial.model_arch}>
                  <td
                    style={{
                      ...cvTdLabelStyle,
                      color: isFirst ? accent : "var(--vf-text-muted)",
                      fontWeight: isFirst ? 700 : 500,
                    }}
                  >
                    #{i + 1}
                    {isFirst && (
                      <span style={{ marginLeft: 6, fontSize: 11 }}>👑</span>
                    )}
                  </td>
                  <td
                    style={{
                      ...cvTdStyle,
                      color: isFirst ? accent : "var(--vf-text)",
                      fontWeight: isFirst ? 600 : 400,
                    }}
                  >
                    {trial.model_arch}
                  </td>
                  <td style={cvTdStyle}>{formatMetric(trial.accuracy)}</td>
                  <td style={cvTdStyle}>{formatMetric(trial.f1)}</td>
                  <td style={cvTdStyle}>{formatMetric(trial.auc_roc)}</td>
                  <td style={cvTdStyle}>
                    {typeof trial.training_time_s === "number"
                      ? trial.training_time_s.toFixed(1)
                      : "—"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: 10,
          color: "var(--vf-text-muted)",
          fontStyle: "italic",
        }}
      >
        <Rich text={t.resultsView.modelComparison.footer} />
      </div>
    </div>
  );
}

function GridStat({
  label,
  value,
  accent,
}: {
  label: string;
  value: string;
  accent: string;
}) {
  return (
    <div
      style={{
        padding: "8px 12px",
        background: "rgba(0,0,0,0.30)",
        border: "1px solid var(--vf-panel-stroke)",
        borderRadius: 8,
      }}
    >
      <div
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: 9,
          letterSpacing: "0.14em",
          textTransform: "uppercase",
          color: "var(--vf-text-muted)",
        }}
      >
        {label}
      </div>
      <div
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: 16,
          fontWeight: 600,
          color: accent,
          marginTop: 2,
        }}
      >
        {value}
      </div>
    </div>
  );
}

function isGridSearchReport(report: Record<string, unknown>): boolean {
  return (
    "best_trial" in report &&
    typeof report["total_trials"] === "number" &&
    typeof report["successful_trials"] === "number"
  );
}

/** Structured render for GridSearchBlock.report().
 *
 * The headline is the winning config — best metric on top, then the
 * hyperparameter overrides that produced it. Total/successful trial counts
 * sit alongside so partial-success runs are visible.
 */
function GridSearchReport({
  report,
  accent,
  submitted = null,
}: {
  report: Record<string, unknown>;
  accent: string;
  submitted?: number | null;
}) {
  const t = useT();
  // Null when a stop landed before any trial finished (ADR-111).
  const best = (report["best_trial"] ?? {}) as Record<string, unknown>;
  const hasBest = report["best_trial"] !== null && report["best_trial"] !== undefined;
  const total = report["total_trials"] as number;
  const successful = report["successful_trials"] as number;
  const plan = unitPlan(total, plannedUnits(report, submitted));

  // Fields written by GridSearchBlock alongside the hyperparameter overrides.
  // Everything else in best_trial is treated as an override and rendered as
  // a key=value chip.
  const META_FIELDS = new Set([
    "trial_index",
    "seed",
    "status",
    "error",
    "best_val_loss",
    "test_accuracy",
    "test_f1",
  ]);
  const overrides = Object.entries(best).filter(
    ([k]) => !META_FIELDS.has(k),
  );

  const metricRow = (label: string, key: string) => {
    const v = best[key];
    if (v === null || v === undefined) return null;
    return (
      <GridStat
        label={label}
        value={
          typeof v === "number" ? (v % 1 === 0 ? String(v) : v.toFixed(4)) : String(v)
        }
        accent={accent}
      />
    );
  };

  return (
    <div style={{ marginTop: 22, display: "flex", flexDirection: "column", gap: 16 }}>
      <div
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: 10,
          letterSpacing: "0.20em",
          textTransform: "uppercase",
          color: "var(--vf-text-muted)",
        }}
      >
        {t.resultsView.gridSearch.title(
          successful,
          plan.total,
          hasBest ? String(best["trial_index"] ?? "?") : "—",
        )}
        {plan.notRun > 0 && t.resultsView.notRun(plan.notRun)}
      </div>

      {hasBest && (
      <div
        style={{
          padding: 16,
          background: `linear-gradient(180deg, ${accent}1c 0%, rgba(12,14,18,0.5) 100%)`,
          border: `1px solid ${accent}55`,
          borderRadius: 12,
          display: "flex",
          flexDirection: "column",
          gap: 14,
        }}
      >
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))",
            gap: 10,
          }}
        >
          {metricRow("test_accuracy", "test_accuracy")}
          {metricRow("test_f1", "test_f1")}
          {metricRow("best_val_loss", "best_val_loss")}
          {metricRow("seed", "seed")}
        </div>

        {overrides.length > 0 && (
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            <div
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: 9,
                letterSpacing: "0.14em",
                textTransform: "uppercase",
                color: "var(--vf-text-muted)",
              }}
            >
              {t.resultsView.gridSearch.overrides}
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
              {overrides.map(([k, v]) => (
                <span
                  key={k}
                  style={{
                    padding: "4px 10px",
                    background: "rgba(0,0,0,0.30)",
                    border: "1px solid var(--vf-panel-stroke)",
                    borderRadius: 8,
                    fontFamily: "var(--font-mono)",
                    fontSize: 11,
                    color: "var(--vf-text)",
                  }}
                >
                  <span style={{ color: "var(--vf-text-muted)" }}>{k}=</span>
                  {String(v)}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
      )}

      <div
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: 10,
          color: "var(--vf-text-muted)",
          fontStyle: "italic",
        }}
      >
        <Rich text={t.resultsView.gridSearch.footer} />
      </div>
    </div>
  );
}

/** The status cell of a unit, for the tables that print a word (ok / failed · why). */
function unitColor(state: ReturnType<typeof unitState>): string {
  if (state === "stopped") return STOPPED_COLOR;
  return state === "ok" ? "var(--vf-text)" : "oklch(0.72 0.19 22)";
}

function unitText(
  t: ReturnType<typeof useT>,
  state: ReturnType<typeof unitState>,
  error: string,
): string {
  if (state === "ok") return t.resultsView.outcome.ok;
  // A unit cut by a stop has no error: the note it carries says why it is out.
  return state === "stopped"
    ? `■ ${t.resultsView.outcome.stopped}`
    : t.resultsView.outcome.failed(error);
}

/** The same, for the tables that print a mark (✓ / × why). */
function markColor(state: ReturnType<typeof unitState>): string {
  if (state === "stopped") return STOPPED_COLOR;
  return state === "ok" ? "oklch(0.85 0.16 150)" : "oklch(0.85 0.14 22)";
}

function markText(
  t: ReturnType<typeof useT>,
  state: ReturnType<typeof unitState>,
  error: string,
): string {
  if (state === "ok") return "✓";
  return state === "stopped"
    ? `■ ${t.resultsView.outcome.stopped}`
    : `× ${error || "?"}`;
}

const cvThStyle: React.CSSProperties = {
  textAlign: "left",
  padding: "8px 10px",
  borderBottom: "1px solid var(--vf-panel-stroke)",
  fontSize: 10,
  letterSpacing: "0.14em",
  textTransform: "uppercase",
  color: "var(--vf-text-muted)",
  fontWeight: 500,
};

const cvTdStyle: React.CSSProperties = {
  padding: "8px 10px",
  borderBottom: "1px solid rgba(255,255,255,0.04)",
  color: "var(--vf-text)",
};

const cvTdLabelStyle: React.CSSProperties = {
  ...cvTdStyle,
  color: "var(--vf-text-muted)",
  fontSize: 11,
  letterSpacing: "0.04em",
};
