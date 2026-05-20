import { artifactUrl } from "../api/client";
import type { RunResult } from "../types/run";

interface ResultsViewProps {
  result: RunResult;
  onClose: () => void;
  taskAccent: string;
}

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
            // gráficos
          </div>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))",
              gap: 16,
            }}
          >
            {graphics.map((path, idx) => (
              <div
                key={idx}
                style={{
                  borderRadius: 12,
                  border: `1px solid ${taskAccent}33`,
                  overflow: "hidden",
                  background: "rgba(0,0,0,0.3)",
                }}
              >
                <img
                  src={artifactUrl(path)}
                  alt={`Plot ${idx + 1}`}
                  style={{ width: "100%", height: "auto", display: "block" }}
                />
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Report summary */}
      {result.report && Object.keys(result.report).length > 0 && (
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
      )}
    </div>
  );
}
