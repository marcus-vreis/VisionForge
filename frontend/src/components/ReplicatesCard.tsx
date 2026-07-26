import { useEffect, useRef, useState } from "react";
import { NumberField, SelectField, Segmented, TextField } from "./controls";
import {
  buildReplicatesPayload,
  parseSeeds,
  seedsProblem,
  type ReplicatesPayload,
  type ReplicatesSeedMode,
} from "../lib/replicates-form";

interface Option {
  value: string;
  label: string;
}

interface ReplicatesCardProps {
  /** Headline metrics; the first is the default (the task's primary metric). */
  metrics: Option[];
  accent: string;
  disabled?: boolean;
  onReplicates: (payload: ReplicatesPayload) => void;
  /** Incremented by the main "Treinar" button so it runs the selected
   *  strategy instead of silently starting a plain single run. */
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

/** Multi-seed replicates launcher (ADR-056): trains the same config N times
 *  under different seeds and reports mean ± 95% CI per metric — the defensible
 *  version of a single-run number. */
export function ReplicatesCard({
  metrics,
  accent,
  disabled,
  onReplicates,
  runSignal,
}: ReplicatesCardProps) {
  const [seedMode, setSeedMode] = useState<ReplicatesSeedMode>("auto");
  const [nReplicates, setNReplicates] = useState(5);
  const [rawSeeds, setRawSeeds] = useState("42, 43, 44, 45, 46");
  const [metric, setMetric] = useState(metrics[0]?.value ?? "");

  const seeds = parseSeeds(rawSeeds);
  const problem = seedMode === "explicit" ? seedsProblem(seeds) : null;
  const count = seedMode === "explicit" ? seeds.length : nReplicates;
  const canRun = !disabled && problem === null && count >= 2;

  const run = () =>
    onReplicates(buildReplicatesPayload(seedMode, nReplicates, rawSeeds, metric));

  // Fire on a new signal only — the card keeps owning its own fields, so the
  // effect deliberately does not depend on them (it would re-fire on typing).
  const lastSignal = useRef(runSignal ?? 0);
  useEffect(() => {
    if (runSignal === undefined || runSignal === lastSignal.current) return;
    lastSignal.current = runSignal;
    if (canRun) run();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runSignal]);

  return (
    <div style={card}>
      <div style={sectionLabel}>Réplicas multi-seed · rigor estatístico</div>
      <p
        style={{
          margin: "0 0 14px",
          fontFamily: "var(--font-mono)",
          fontSize: 11,
          lineHeight: 1.6,
          color: "var(--vf-text-muted)",
        }}
      >
        Treina a mesma config N vezes sob seeds diferentes e agrega cada métrica
        em média ± IC 95% (t de Student). Um run único é uma amostra de uma
        distribuição — réplicas tornam o número defensável.
      </p>

      <div style={{ display: "flex", gap: 12, alignItems: "flex-end", flexWrap: "wrap" }}>
        <div style={{ minWidth: 220 }}>
          <Segmented
            label="Seeds"
            value={seedMode}
            onChange={(v) => setSeedMode(v as ReplicatesSeedMode)}
            options={[
              { value: "auto", label: "Automáticas" },
              { value: "explicit", label: "Explícitas" },
            ]}
            hint={
              seedMode === "auto"
                ? "consecutivas a partir do training.seed"
                : "lista exata, reproduzível"
            }
          />
        </div>
        {seedMode === "auto" ? (
          <div style={{ width: 130 }}>
            <NumberField
              label="Nº de réplicas"
              value={nReplicates}
              onChange={(v) => setNReplicates(Math.min(50, Math.max(2, Math.round(v))))}
              min={2}
              max={50}
              step={1}
            />
          </div>
        ) : (
          <div style={{ flex: "1 1 240px" }}>
            <TextField
              label="Seeds (vírgula)"
              value={rawSeeds}
              onChange={setRawSeeds}
              placeholder="42, 43, 44, 45, 46"
              hint={problem ?? `${seeds.length} seeds`}
              mono
            />
          </div>
        )}
        <div style={{ minWidth: 170 }}>
          <SelectField
            label="Métrica destaque"
            value={metric}
            onChange={setMetric}
            options={metrics}
          />
        </div>
        <button
          type="button"
          onClick={run}
          disabled={!canRun}
          style={{
            padding: "12px 18px",
            background: canRun ? accent : "rgba(255,255,255,0.05)",
            border: `1px solid ${canRun ? accent : "var(--vf-panel-stroke)"}`,
            borderRadius: 10,
            color: canRun ? "var(--accent-ink, #08120c)" : "var(--vf-text-muted)",
            fontFamily: "var(--font-mono)",
            fontSize: 12,
            fontWeight: 600,
            letterSpacing: "0.04em",
            cursor: canRun ? "pointer" : "not-allowed",
            whiteSpace: "nowrap",
          }}
        >
          🎲 Rodar réplicas · {count} seed{count === 1 ? "" : "s"}
        </button>
      </div>
    </div>
  );
}
