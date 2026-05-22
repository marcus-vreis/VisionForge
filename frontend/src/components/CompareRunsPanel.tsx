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
    setLoading(true);
    Promise.all(runIds.map((id) => fetchRunDetail(id)))
      .then((arr) => {
        if (alive) setDetails(arr);
      })
      .catch((e: unknown) => {
        if (alive) {
          setError(e instanceof Error ? e.message : "Falha ao carregar runs.");
        }
      })
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
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
