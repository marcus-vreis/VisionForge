import { useState } from "react";
import { artifactUrl, downloadRunMarkdown } from "../api/client";
import type { RunResult } from "../types/run";
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
};

interface MetricCardProps {
  label: string;
  value: string;
  accent: string;
  highlight?: boolean;
}

function MetricCard({ label, value, accent, highlight }: MetricCardProps) {
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

interface ModelComparisonTrial {
  model_arch: string;
  status: string;
  error?: string;
  training_time_s: number | null;
  // Task-specific metric columns (accuracy/f1/auc_roc, mse/rmse/mae/r2,
  // miou/dice/pixel_acc, …) — metric-agnostic so one renderer serves every task.
  [metric: string]: number | string | null | undefined;
}

// Structural keys present on every trial regardless of task — everything else
// is a metric column to be rendered dynamically.
const COMPARISON_STRUCTURAL_KEYS = ["model_arch", "status", "error", "training_time_s"];

// Friendly column headers for known metrics; unknown keys fall back to the raw
// key so a new task's metric still renders without a code change here.
const COMPARISON_METRIC_LABELS: Record<string, string> = {
  accuracy: "Accuracy",
  f1: "F1",
  auc_roc: "AUC-ROC",
  mse: "MSE",
  rmse: "RMSE",
  mae: "MAE",
  r2: "R²",
  miou: "mIoU",
  dice: "Dice",
  pixel_acc: "Pixel acc",
};

function isModelComparisonReport(report: Record<string, unknown>): boolean {
  return (
    Array.isArray(report["top_3"]) &&
    typeof report["total_ran"] === "number" &&
    typeof report["failed_count"] === "number"
  );
}

/** Structured render for a model-comparison report (any task, ADR-041).
 *
 * Surfaces the ranked top-3 architectures plus run totals. Metric columns are
 * derived from the trial objects, so classification (accuracy/f1/auc_roc),
 * regression (mse/rmse/mae/r2) and segmentation (miou/dice/pixel_acc) all
 * render through one component.
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

  const metricKeys =
    top3.length > 0
      ? Object.keys(top3[0]).filter(
          (k) => !COMPARISON_STRUCTURAL_KEYS.includes(k),
        )
      : [];

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
              {metricKeys.map((k) => (
                <th key={k} style={cvThStyle}>
                  {COMPARISON_METRIC_LABELS[k] ?? k}
                </th>
              ))}
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
                  {metricKeys.map((k) => (
                    <td key={k} style={cvTdStyle}>
                      {formatMetric(trial[k])}
                    </td>
                  ))}
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
