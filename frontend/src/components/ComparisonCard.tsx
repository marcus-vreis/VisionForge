import { useState } from "react";
import { SelectField } from "./controls";

interface Option {
  value: string;
  label: string;
}

interface ComparisonCardProps {
  /** Architectures the user can pick from (the task's model list). */
  modelOptions: Option[];
  /** Ranking metrics; the first is the default (the task's primary metric). */
  metrics: Option[];
  accent: string;
  disabled?: boolean;
  onCompare: (modelNames: string[], metric: string) => void;
}

const card: React.CSSProperties = {
  background: "var(--vf-panel)",
  border: "1px solid var(--vf-panel-stroke)",
  borderRadius: 18,
  padding: 26,
  backdropFilter: "blur(14px)",
};

const sectionLabel: React.CSSProperties = {
  fontFamily: "var(--font-mono)",
  fontSize: 10,
  letterSpacing: "0.22em",
  textTransform: "uppercase",
  color: "var(--vf-text-muted)",
  marginBottom: 12,
};

/** Advanced "compare N architectures" surface for a standalone task (ADR-044).
 *  Trains each picked arch on the same dataset and ranks by the chosen metric. */
export function ComparisonCard({
  modelOptions,
  metrics,
  accent,
  disabled,
  onCompare,
}: ComparisonCardProps) {
  const [selected, setSelected] = useState<string[]>([]);
  const [metric, setMetric] = useState(metrics[0]?.value ?? "");

  const toggle = (value: string) =>
    setSelected((prev) =>
      prev.includes(value)
        ? prev.filter((v) => v !== value)
        : [...prev, value],
    );

  const canCompare = selected.length >= 2 && !disabled;

  return (
    <div style={card}>
      <div style={sectionLabel}>Comparar arquiteturas · modo avançado</div>
      <p
        style={{
          margin: "0 0 14px",
          fontFamily: "var(--font-mono)",
          fontSize: 11,
          lineHeight: 1.6,
          color: "var(--vf-text-muted)",
        }}
      >
        Treina cada arquitetura marcada no mesmo dataset e ranqueia pela métrica
        escolhida (mín. 2).
      </p>

      <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginBottom: 16 }}>
        {modelOptions.map((o) => {
          const on = selected.includes(o.value);
          return (
            <button
              key={o.value}
              type="button"
              onClick={() => toggle(o.value)}
              style={{
                padding: "7px 12px",
                background: on ? "var(--accent-soft)" : "rgba(255,255,255,0.03)",
                border: `1px solid ${on ? accent : "var(--vf-panel-stroke)"}`,
                borderRadius: 999,
                fontFamily: "var(--font-mono)",
                fontSize: 11,
                color: on ? "var(--vf-text)" : "var(--vf-text-dim)",
                cursor: "pointer",
              }}
            >
              {on ? "✓ " : ""}
              {o.label}
            </button>
          );
        })}
      </div>

      <div style={{ display: "flex", gap: 12, alignItems: "flex-end", flexWrap: "wrap" }}>
        <div style={{ minWidth: 200 }}>
          <SelectField
            label="Métrica de ranking"
            value={metric}
            onChange={setMetric}
            options={metrics}
          />
        </div>
        <button
          type="button"
          onClick={() => onCompare(selected, metric)}
          disabled={!canCompare}
          style={{
            padding: "12px 18px",
            background: canCompare ? accent : "rgba(255,255,255,0.05)",
            border: `1px solid ${canCompare ? accent : "var(--vf-panel-stroke)"}`,
            borderRadius: 10,
            color: canCompare ? "var(--accent-ink, #08120c)" : "var(--vf-text-muted)",
            fontFamily: "var(--font-mono)",
            fontSize: 12,
            fontWeight: 600,
            letterSpacing: "0.04em",
            cursor: canCompare ? "pointer" : "not-allowed",
            whiteSpace: "nowrap",
          }}
        >
          ⛓ Comparar {selected.length} arquitetura{selected.length === 1 ? "" : "s"}
        </button>
      </div>
    </div>
  );
}
