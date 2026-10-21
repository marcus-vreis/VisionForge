import { useEffect, useRef, useState } from "react";
import { RunSummary } from "./RunSummary";
import { useT } from "../i18n/useT";
import { SelectField } from "./controls";
import {
  buildComparePayload,
  toggleModel,
  type ComparePayload,
} from "../lib/compare-form";

interface Option {
  value: string;
  label: string;
}

interface CompareModelsCardProps {
  /** The architectures to pick from; all start selected. */
  models: Option[];
  /** Ranking metrics; the first is the default (the task's primary metric). */
  metrics: Option[];
  /** Says which split the ranking reads and that it is a report (ADR-131). */
  note: string;
  accent: string;
  disabled?: boolean;
  onCompare: (payload: ComparePayload) => void;
  /** Incremented by the main "Treinar" button so it runs this card. */
  runSignal?: number;
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

/** Architecture comparison launcher (ADR-044): trains each picked model on the
 *  same config and ranks them by one metric. */
export function CompareModelsCard({
  models,
  metrics,
  note,
  accent,
  disabled,
  onCompare,
  runSignal,
}: CompareModelsCardProps) {
  const t = useT();
  const order = models.map((m) => m.value);
  const [selected, setSelected] = useState<string[]>(order);
  const [metric, setMetric] = useState(metrics[0]?.value ?? "");

  const canRun = !disabled && selected.length >= 2;
  const run = () => onCompare(buildComparePayload(selected, metric));

  // Fire on a new signal only, as ReplicatesCard does.
  const [attempted, setAttempted] = useState(false);
  const lastSignal = useRef(runSignal ?? 0);
  useEffect(() => {
    if (runSignal === undefined || runSignal === lastSignal.current) return;
    lastSignal.current = runSignal;
    if (canRun) run();
    else setAttempted(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runSignal]);

  return (
    <div style={card}>
      <div style={sectionLabel}>{t.compareModelsCard.title}</div>
      <p
        style={{
          margin: "0 0 14px",
          fontFamily: "var(--font-mono)",
          fontSize: 11,
          lineHeight: 1.6,
          color: "var(--vf-text-muted)",
        }}
      >
        {note}
      </p>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))",
          gap: 8,
          marginBottom: 14,
        }}
      >
        {models.map((m) => {
          const on = selected.includes(m.value);
          return (
            <label
              key={m.value}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                padding: "8px 12px",
                background: on ? "var(--accent-soft)" : "rgb(var(--lt-shade, 0 0 0) / 0.30)",
                border: `1px solid ${on ? "var(--accent-vf)" : "var(--vf-panel-stroke)"}`,
                borderRadius: 8,
                fontFamily: "var(--font-mono)",
                fontSize: 12,
                color: on ? "var(--vf-text)" : "var(--vf-text-dim)",
                cursor: "pointer",
              }}
            >
              <input
                type="checkbox"
                checked={on}
                onChange={() => setSelected((s) => toggleModel(s, m.value, order))}
                style={{ accentColor: "var(--accent-vf)" }}
              />
              {m.label}
            </label>
          );
        })}
      </div>

      <div style={{ display: "flex", gap: 12, alignItems: "flex-end", flexWrap: "wrap" }}>
        <div style={{ minWidth: 170 }}>
          <SelectField
            label={t.compareModelsCard.metric}
            value={metric}
            onChange={setMetric}
            options={metrics}
          />
        </div>
        <RunSummary
          text={t.compareModelsCard.summary(selected.length)}
          ready={canRun}
          reason={disabled ? null : t.compareModelsCard.blocked}
          attempted={attempted}
          accent={accent}
        />
      </div>
    </div>
  );
}
