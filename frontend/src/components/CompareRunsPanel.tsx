import { useEffect, useState } from "react";
import { fetchRunDetail, type RunDetail } from "../api/client";

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

const METRIC_ROWS: Array<{ key: string; label: string }> = [
  { key: "best_val_loss", label: "Melhor val loss" },
  { key: "best_epoch", label: "Melhor epoch" },
  { key: "total_epochs", label: "Total de epochs" },
  { key: "test_accuracy", label: "Acurácia (teste)" },
  { key: "test_f1", label: "F1 (teste)" },
  { key: "test_precision", label: "Precisão (teste)" },
  { key: "test_recall", label: "Recall (teste)" },
  { key: "test_auc_roc", label: "AUC-ROC (teste)" },
];

function fmtMetric(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") return v % 1 === 0 ? String(v) : v.toFixed(4);
  return String(v);
}

/** Side-by-side metric + overlaid epoch-curve view for 2+ historical runs. */
export function CompareRunsPanel({ runIds, onBack }: CompareRunsPanelProps) {
  const [details, setDetails] = useState<RunDetail[]>([]);
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
      .then((arr) => {
        if (alive) setDetails(arr);
      })
      .catch((e: unknown) => {
        if (alive) {
          setError(e instanceof Error ? e.message : "Falha ao carregar runs.");
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
          ← histórico
        </button>
        <div style={{ fontFamily: "var(--font-mono)", fontSize: 14, color: "var(--vf-text)" }}>
          Comparando {runIds.length} runs
        </div>
      </div>

      {loading && (
        <div style={{ padding: 32, textAlign: "center", color: "var(--vf-text-muted)" }}>
          carregando runs…
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

      {!loading && !error && details.length > 0 && (
        <>
          <Legend details={details} />
          <MetricsTable details={details} />
          <ConfigDiffTable details={details} />
          <PreprocessingCompare details={details} />
          <OverlayChart
            details={details}
            yKey="val_loss"
            title="Val loss × epoch"
            invertGood
          />
          <OverlayChart
            details={details}
            yKey="val_accuracy"
            title="Val accuracy × epoch"
          />
        </>
      )}
    </div>
  );
}

function Legend({ details }: { details: RunDetail[] }) {
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
        </div>
      ))}
    </div>
  );
}

function MetricsTable({ details }: { details: RunDetail[] }) {
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
            <th style={thStyle}>Métrica</th>
            {details.map((d, i) => (
              <th key={d.run_id} style={{ ...thStyle, color: PALETTE[i % PALETTE.length] }}>
                {d.experiment_name}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {METRIC_ROWS.map(({ key, label }) => {
            const present = details.some((d) => d.metrics[key] !== undefined);
            if (!present) return null;
            return (
              <tr key={key}>
                <td style={tdLabelStyle}>{label}</td>
                {details.map((d) => (
                  <td key={d.run_id} style={tdStyle}>
                    {fmtMetric(d.metrics[key])}
                  </td>
                ))}
              </tr>
            );
          })}
          <tr>
            <td style={tdLabelStyle}>Dispositivo</td>
            {details.map((d) => (
              <td key={d.run_id} style={tdStyle}>
                {d.device_used ?? "—"}
              </td>
            ))}
          </tr>
        </tbody>
      </table>
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

const tdLabelStyle: React.CSSProperties = {
  ...tdStyle,
  color: "var(--vf-text-muted)",
  fontSize: 11,
  letterSpacing: "0.04em",
};

// ── Config diff ──────────────────────────────────────────────────────────────

/** Selectors that pull comparable scalar values from RunDetail.config. */
const CONFIG_ROWS: Array<{ label: string; pick: (cfg: Record<string, unknown>) => unknown }> = [
  { label: "Arquitetura", pick: (c) => (c.model as Record<string, unknown> | undefined)?.name },
  { label: "Num classes", pick: (c) => (c.model as Record<string, unknown> | undefined)?.num_classes },
  { label: "Pretrained", pick: (c) => (c.model as Record<string, unknown> | undefined)?.pretrained },
  { label: "Task", pick: (c) => c.task },
  { label: "Learning rate", pick: (c) => (c.training as Record<string, unknown> | undefined)?.learning_rate },
  { label: "Optimizer", pick: (c) => (c.training as Record<string, unknown> | undefined)?.optimizer },
  { label: "Batch size", pick: (c) => (c.training as Record<string, unknown> | undefined)?.batch_size },
  { label: "Epochs (max)", pick: (c) => (c.training as Record<string, unknown> | undefined)?.epochs },
  { label: "Weight decay", pick: (c) => (c.training as Record<string, unknown> | undefined)?.weight_decay },
  { label: "Seed", pick: (c) => (c.training as Record<string, unknown> | undefined)?.seed },
  { label: "Mixed precision", pick: (c) => (c.training as Record<string, unknown> | undefined)?.mixed_precision },
  {
    label: "Scheduler",
    pick: (c) => {
      const s = (c.training as Record<string, unknown> | undefined)?.scheduler as
        | Record<string, unknown>
        | undefined;
      return s?.kind ?? "none";
    },
  },
  { label: "Image size", pick: (c) => pickDataTransforms(c)?.image_size },
  { label: "Horizontal flip", pick: (c) => pickDataTransforms(c)?.horizontal_flip },
  { label: "Rotation (°)", pick: (c) => pickDataTransforms(c)?.rotation_degrees },
  { label: "Color jitter", pick: (c) => pickDataTransforms(c)?.color_jitter },
  {
    label: "Preprocessing",
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
        // diff de configuração (células destacadas = diferentes da 1ª run)
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
            <th style={thStyle}>Campo</th>
            {details.map((d, i) => (
              <th key={d.run_id} style={{ ...thStyle, color: PALETTE[i % PALETTE.length] }}>
                {d.experiment_name}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {CONFIG_ROWS.map(({ label, pick }) => {
            const values = details.map((d) => pick(d.config));
            const reference = values[0];
            const anyDiff = values.some((v) => !sameConfigValue(v, reference));
            return (
              <tr key={label}>
                <td style={tdLabelStyle}>{label}</td>
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
        // pipelines de pré-processamento
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
                  sem pré-processamento
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

interface OverlayChartProps {
  details: RunDetail[];
  yKey: "val_loss" | "val_accuracy" | "train_loss" | "train_accuracy";
  title: string;
  invertGood?: boolean;
}

function OverlayChart({ details, yKey, title }: OverlayChartProps) {
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
        y: (h as Record<string, number>)[yKey] ?? NaN,
      }));
      return { runId: d.run_id, label: d.experiment_name, color: PALETTE[i % PALETTE.length], points };
    })
    .filter((s) => s.points.length > 0);

  if (series.length === 0) {
    return null;
  }

  const allX = series.flatMap((s) => s.points.map((p) => p.x));
  const allY = series.flatMap((s) => s.points.map((p) => p.y)).filter((v) => Number.isFinite(v));
  const xMax = Math.max(...allX);
  const yMin = Math.min(...allY);
  const yMax = Math.max(...allY);
  const yRange = yMax - yMin || 1;

  const xScale = (x: number) => padding.left + (x / Math.max(xMax, 1)) * innerW;
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
        // {title}
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
          1
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
          const path = s.points
            .filter((p) => Number.isFinite(p.y))
            .map((p, i) => `${i === 0 ? "M" : "L"} ${xScale(p.x)} ${yScale(p.y)}`)
            .join(" ");
          return (
            <path
              key={s.runId}
              d={path}
              fill="none"
              stroke={s.color}
              strokeWidth="2"
              strokeLinejoin="round"
              strokeLinecap="round"
              opacity={0.92}
            />
          );
        })}
      </svg>
    </div>
  );
}
