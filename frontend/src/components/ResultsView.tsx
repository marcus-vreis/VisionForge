import { useState } from "react";
import { artifactUrl, downloadRunMarkdown } from "../api/client";
import { metricCi } from "../lib/metric-ci";
import type { MetricCI, RunResult } from "../types/run";
import { Lightbox } from "./Lightbox";

interface ResultsViewProps {
  result: RunResult;
  onClose: () => void;
  taskAccent: string;
}

/** Plot file naming convention from the backend — kept in sync with
 * MetricsPlotter so users see human labels instead of raw filenames. */
const GRAPH_LABELS: Record<string, string> = {
  "loss.png": "Loss (train + val)",
  "accuracy.png": "Accuracy (train + val)",
  "confusion_matrix.png": "Matriz de confusão",
  "confusion_matrix_normalized.png": "Matriz de confusão (normalizada)",
  "roc_curve.png": "Curva ROC",
  "precision_recall_curve.png": "Curva Precision-Recall",
  // Detection (Ultralytics) plot names.
  "results.png": "Curvas de treino (Ultralytics)",
  "BoxPR_curve.png": "Curva Precision-Recall (box)",
  "BoxF1_curve.png": "Curva F1 (box)",
  // Test-set diagnostics per task (ADR-077).
  "auroc.png": "AUROC por época",
  "pred_vs_true.png": "Predito vs real",
  "residuals.png": "Distribuição dos resíduos",
  "iou_per_class.png": "IoU por classe",
  "score_histogram.png": "Escores: normal vs defeito",
};

/** Format metric values for display. */
function formatMetric(value: unknown): string {
  if (value === null || value === undefined) return "N/A";
  if (typeof value === "number") {
    return value % 1 === 0 ? String(value) : value.toFixed(4);
  }
  return String(value);
}

/** Human-readable metric labels. */
const METRIC_LABELS: Record<string, string> = {
  best_val_loss: "Best Val Loss",
  best_epoch: "Best Epoch",
  total_epochs: "Total Epochs",
  test_accuracy: "Accuracy",
  test_f1: "F1 Score",
  test_precision: "Precision",
  test_recall: "Recall",
  test_auc_roc: "AUC-ROC",
  // Detection metrics.
  map50: "mAP@50",
  map50_95: "mAP@50-95",
  precision: "Precision (box)",
  recall: "Recall (box)",
  box_loss: "Box loss (val)",
};

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
  return (
    <div
      title={
        `IC ${Math.round(ci.confidence * 100)}% por bootstrap percentil: ` +
        `${ci.n_resamples} reamostragens das ${ci.n_samples} imagens de teste. ` +
        `Mede o ruído de amostragem do split com este modelo fixo — não a ` +
        `variação entre treinos, que réplicas com várias seeds medem.`
      }
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
export function ResultsView({ result, onClose, taskAccent }: ResultsViewProps) {
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
            // resultados
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
          title="Baixar model card (markdown) deste run"
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
          ↓ markdown
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
              label={METRIC_LABELS[key] ?? key}
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
            // gráficos · clique para expandir
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
              const label = GRAPH_LABELS[filename] ?? filename;
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
          <CrossValidationReport report={result.report} accent={taskAccent} />
        ) : isTaskCvReport(result.report) ? (
          <TaskCvReport report={result.report} accent={taskAccent} />
        ) : isReplicatesReport(result.report) ? (
          <ReplicatesReport report={result.report} accent={taskAccent} />
        ) : isTaskComparisonReport(result.report) ? (
          <TaskComparisonReport report={result.report} accent={taskAccent} />
        ) : isTaskSweepReport(result.report) ? (
          <TaskSweepReport report={result.report} accent={taskAccent} />
        ) : isModelComparisonReport(result.report) ? (
          <ModelComparisonReport report={result.report} accent={taskAccent} />
        ) : isGridSearchReport(result.report) ? (
          <GridSearchReport report={result.report} accent={taskAccent} />
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
              // report
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

function isCrossValidationReport(report: Record<string, unknown>): boolean {
  return (
    Array.isArray(report["fold_results"]) &&
    typeof report["mean_accuracy"] === "number" &&
    typeof report["std_accuracy"] === "number"
  );
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
}: {
  report: Record<string, unknown>;
  accent: string;
}) {
  const folds = (report["fold_results"] as FoldRecord[]) ?? [];
  const meanAcc = report["mean_accuracy"] as number;
  const stdAcc = report["std_accuracy"] as number;
  const meanF1 = report["mean_f1"] as number;
  const stdF1 = report["std_f1"] as number;

  const successful = folds.filter((f) => f.status === "success");
  const failed = folds.filter((f) => f.status !== "success");

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
        // k-fold cross-validation · {successful.length}/{folds.length} folds ok
        {failed.length > 0 ? ` · ${failed.length} falharam` : ""}
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
          label="Acurácia (média ± std)"
          mean={meanAcc}
          std={stdAcc}
          accent={accent}
          highlight
        />
        <AggregateCard
          label="F1 (média ± std)"
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
              <th style={cvThStyle}>Fold</th>
              <th style={cvThStyle}>train_size</th>
              <th style={cvThStyle}>val_size</th>
              <th style={cvThStyle}>val_loss</th>
              <th style={cvThStyle}>accuracy</th>
              <th style={cvThStyle}>F1</th>
              <th style={cvThStyle}>status</th>
            </tr>
          </thead>
          <tbody>
            {folds.map((f) => {
              const ok = f.status === "success";
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
                      color: ok
                        ? "oklch(0.85 0.16 150)"
                        : "oklch(0.85 0.14 22)",
                    }}
                  >
                    {ok ? "✓" : `× ${f.error || "?"}`}
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
  mean: number;
  std: number;
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
        {mean.toFixed(4)}
        <span
          style={{
            fontSize: 14,
            color: "var(--vf-text-muted)",
            marginLeft: 6,
            fontWeight: 400,
          }}
        >
          ± {std.toFixed(4)}
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
}: {
  report: Record<string, unknown>;
  accent: string;
}) {
  const folds = (report["fold_results"] as TaskCvFoldRow[]) ?? [];
  const aggregate =
    (report["aggregate"] as Record<string, { mean: number; std: number }>) ?? {};
  const metric = report["metric"] as string;
  const nFolds = report["n_folds"] as number;
  const ok = report["successful_folds"] as number;
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
        // k-fold · {ok}/{nFolds} folds ok · destaque {metric}
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
            média ± desvio sobre os folds
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
              <th style={cvThStyle}>fold</th>
              {metricKeys.map((k) => (
                <th key={k} style={cvThStyle}>
                  {k}
                </th>
              ))}
              <th style={cvThStyle}>treino/val</th>
              <th style={cvThStyle}>status</th>
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
                    color: f.status === "success" ? "var(--vf-text)" : "oklch(0.72 0.19 22)",
                  }}
                >
                  {f.status === "success" ? "ok" : `falhou · ${f.error}`}
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
}: {
  report: Record<string, unknown>;
  accent: string;
}) {
  const trials = (report["trials"] as ReplicateTrialRow[]) ?? [];
  const metric = report["metric"] as string;
  const aggregates =
    (report["aggregates"] as Record<string, ReplicateAggregate>) ?? {};
  const headline = report["headline"] as ReplicateAggregate | null;
  const total = report["total_replicates"] as number;
  const ok = report["successful_replicates"] as number;

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
        // réplicas multi-seed · {ok}/{total} seeds ok · destaque {metric}
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
            🎯 resultado citável
          </div>
          <div style={{ fontSize: 20, color: "var(--vf-text)" }}>
            {metric} = <span style={{ color: accent }}>{formatMetric(headline.mean)}</span>
            {ciHalf !== null && (
              <span style={{ color: "var(--vf-text-dim)" }}> ± {formatMetric(ciHalf)}</span>
            )}
            <span style={{ fontSize: 11, color: "var(--vf-text-muted)", marginLeft: 10 }}>
              {ciHalf !== null ? "IC 95% · " : ""}n={headline.n}
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
              <th style={cvThStyle}>métrica</th>
              <th style={cvThStyle}>n</th>
              <th style={cvThStyle}>média</th>
              <th style={cvThStyle}>desvio</th>
              <th style={cvThStyle}>min</th>
              <th style={cvThStyle}>max</th>
              <th style={cvThStyle}>IC 95%</th>
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
              <th style={cvThStyle}>seed</th>
              <th style={cvThStyle}>{metric}</th>
              <th style={cvThStyle}>tempo (s)</th>
              <th style={cvThStyle}>status</th>
            </tr>
          </thead>
          <tbody>
            {trials.map((t) => (
              <tr key={t.seed}>
                <td style={cvTdLabelStyle}>{t.seed}</td>
                <td style={cvTdStyle}>{formatMetric(t.metrics?.[metric])}</td>
                <td style={cvTdStyle}>
                  {t.training_time_s === null ? "—" : t.training_time_s.toFixed(1)}
                </td>
                <td
                  style={{
                    ...cvTdStyle,
                    color: t.status === "success" ? "var(--vf-text)" : "oklch(0.72 0.19 22)",
                  }}
                >
                  {t.status === "success" ? "ok" : `falhou · ${t.error}`}
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
}: {
  report: Record<string, unknown>;
  accent: string;
}) {
  const trials = (report["trials"] as TaskComparisonTrial[]) ?? [];
  const metric = report["metric"] as string;
  const totalRan = report["total_ran"] as number;
  const failedCount = report["failed_count"] as number;

  const successful = trials.filter((t) => t.status === "success");
  const otherKeys = Array.from(
    new Set(successful.flatMap((t) => Object.keys(t.metrics ?? {}))),
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
        // comparação de arquiteturas · {totalRan - failedCount}/{totalRan} ok
        {failedCount > 0 ? ` · ${failedCount} falharam` : ""} · ranking por {metric}
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
              <th style={cvThStyle}>Rank</th>
              <th style={cvThStyle}>Arquitetura</th>
              {metricCols.map((k) => (
                <th key={k} style={cvThStyle}>
                  {k}
                </th>
              ))}
              <th style={cvThStyle}>tempo (s)</th>
              <th style={cvThStyle}>status</th>
            </tr>
          </thead>
          <tbody>
            {trials.map((trial, i) => {
              const ok = trial.status === "success";
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
                    {trial.training_time_s !== null &&
                    trial.training_time_s !== undefined
                      ? trial.training_time_s.toFixed(1)
                      : "—"}
                  </td>
                  <td
                    style={{
                      ...cvTdStyle,
                      color: ok ? "oklch(0.85 0.16 150)" : "oklch(0.85 0.14 22)",
                    }}
                  >
                    {ok ? "✓" : `× ${trial.error || "?"}`}
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
}: {
  report: Record<string, unknown>;
  accent: string;
}) {
  const trials = (report["trials"] as TaskSweepTrial[]) ?? [];
  const mode = report["mode"] as string;
  const metric = report["metric"] as string;
  const total = report["total_trials"] as number;
  const successful = report["successful_trials"] as number;
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
        // sweep {mode} · {successful}/{total} trials ok · ranking por {metric}
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
            👑 melhor trial · {metric}=
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
              <th style={cvThStyle}>Rank</th>
              <th style={cvThStyle}>{metric}</th>
              <th style={cvThStyle}>overrides</th>
              <th style={cvThStyle}>tempo (s)</th>
              <th style={cvThStyle}>status</th>
            </tr>
          </thead>
          <tbody>
            {trials.map((trial, i) => {
              const ok = trial.status === "success";
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
                    {trial.training_time_s !== null &&
                    trial.training_time_s !== undefined
                      ? trial.training_time_s.toFixed(1)
                      : "—"}
                  </td>
                  <td
                    style={{
                      ...cvTdStyle,
                      color: ok ? "oklch(0.85 0.16 150)" : "oklch(0.85 0.14 22)",
                    }}
                  >
                    {ok ? "✓" : `× ${trial.error || "?"}`}
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
}: {
  report: Record<string, unknown>;
  accent: string;
}) {
  const top3 = (report["top_3"] as ModelComparisonTrial[]) ?? [];
  const totalRan = report["total_ran"] as number;
  const failedCount = report["failed_count"] as number;

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
        // comparação de modelos · {totalRan - failedCount}/{totalRan} ok
        {failedCount > 0 ? ` · ${failedCount} falharam` : ""}
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
              <th style={cvThStyle}>Rank</th>
              <th style={cvThStyle}>Arquitetura</th>
              <th style={cvThStyle}>Accuracy</th>
              <th style={cvThStyle}>F1</th>
              <th style={cvThStyle}>AUC-ROC</th>
              <th style={cvThStyle}>tempo (s)</th>
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
                    {trial.training_time_s !== null
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
        Top-3 acima. O ranking completo está em
        <code style={{ marginLeft: 6 }}>
          outputs/reports/&lt;experiment&gt;/ranking.csv
        </code>
        .
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
}: {
  report: Record<string, unknown>;
  accent: string;
}) {
  const best = (report["best_trial"] ?? {}) as Record<string, unknown>;
  const total = report["total_trials"] as number;
  const successful = report["successful_trials"] as number;

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
        // grid search · {successful}/{total} trials ok · 👑 melhor trial #
        {String(best["trial_index"] ?? "?")}
      </div>

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
              // overrides do trial vencedor
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

      <div
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: 10,
          color: "var(--vf-text-muted)",
          fontStyle: "italic",
        }}
      >
        Tabela completa em <code>outputs/reports/&lt;experiment&gt;/grid_search_summary.csv</code>
        · config vencedora em <code>best_config.yaml</code>.
      </div>
    </div>
  );
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
