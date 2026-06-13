import { useState } from "react";
import { NumberField, SelectField, Segmented, TextField } from "./controls";
import {
  buildSearchSpace,
  gridTrialCount,
  makeSweepRow,
  type RandomKind,
  type SweepMode,
  type SweepRow,
} from "../lib/sweep-space";

interface Option {
  value: string;
  label: string;
}

export interface SweepPayload {
  mode: SweepMode;
  search_space: Record<string, unknown>;
  metric: string;
  n_trials: number;
  seed: number;
}

interface SweepCardProps {
  /** Ranking metrics; the first is the default (the task's primary metric). */
  metrics: Option[];
  /** Suggested dot-paths shown as a hint (e.g. training.learning_rate). */
  pathHints: string[];
  accent: string;
  disabled?: boolean;
  onSweep: (payload: SweepPayload) => void;
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

const RANDOM_KINDS: Option[] = [
  { value: "uniform", label: "Uniforme" },
  { value: "log_uniform", label: "Log-uniforme" },
  { value: "choice", label: "Escolha" },
];

/** Advanced grid/random hyperparameter sweep editor for a standalone task
 *  (ADR-045). Builds the backend `search_space` from dot-path rows. */
export function SweepCard({
  metrics,
  pathHints,
  accent,
  disabled,
  onSweep,
}: SweepCardProps) {
  const [mode, setMode] = useState<SweepMode>("grid");
  const [rows, setRows] = useState<SweepRow[]>([makeSweepRow()]);
  const [metric, setMetric] = useState(metrics[0]?.value ?? "");
  const [nTrials, setNTrials] = useState(10);
  const [seed, setSeed] = useState(0);

  const patchRow = (i: number, patch: Partial<SweepRow>) =>
    setRows((prev) => prev.map((r, idx) => (idx === i ? { ...r, ...patch } : r)));
  const addRow = () => setRows((prev) => [...prev, makeSweepRow()]);
  const removeRow = (i: number) =>
    setRows((prev) => (prev.length > 1 ? prev.filter((_, idx) => idx !== i) : prev));

  const searchSpace = buildSearchSpace(mode, rows);
  const paramCount = Object.keys(searchSpace).length;
  const trialCount = mode === "grid" ? gridTrialCount(rows) : nTrials;
  const canRun = paramCount > 0 && trialCount > 0 && !disabled;

  return (
    <div style={card}>
      <div style={sectionLabel}>Sweep de hiperparâmetros · modo avançado</div>
      <p
        style={{
          margin: "0 0 14px",
          fontFamily: "var(--font-mono)",
          fontSize: 11,
          lineHeight: 1.6,
          color: "var(--vf-text-muted)",
        }}
      >
        Varre hiperparâmetros por dot-path (ex.: {pathHints.join(", ")}) e ranqueia
        pela métrica. Grid = produto cartesiano; Random = amostras.
      </p>

      <div style={{ maxWidth: 320, marginBottom: 16 }}>
        <Segmented
          label="Estratégia"
          value={mode}
          onChange={(v) => setMode(v as SweepMode)}
          options={[
            { value: "grid", label: "Grid" },
            { value: "random", label: "Random" },
          ]}
        />
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 12, marginBottom: 14 }}>
        {rows.map((row, i) => (
          <div
            key={i}
            style={{
              display: "flex",
              gap: 10,
              alignItems: "flex-end",
              flexWrap: "wrap",
              padding: "10px 12px",
              background: "rgba(0,0,0,0.22)",
              border: "1px solid var(--vf-panel-stroke)",
              borderRadius: 10,
            }}
          >
            <div style={{ flex: "1 1 200px" }}>
              <TextField
                label="Parâmetro (dot-path)"
                value={row.path}
                onChange={(v) => patchRow(i, { path: v })}
                placeholder="training.learning_rate"
                mono
              />
            </div>

            {mode === "grid" ? (
              <div style={{ flex: "1 1 220px" }}>
                <TextField
                  label="Valores (vírgula)"
                  value={row.values}
                  onChange={(v) => patchRow(i, { values: v })}
                  placeholder="0.001, 0.01, 0.1"
                  mono
                />
              </div>
            ) : (
              <>
                <div style={{ flex: "0 0 150px" }}>
                  <SelectField
                    label="Distribuição"
                    value={row.kind}
                    onChange={(v) => patchRow(i, { kind: v as RandomKind })}
                    options={RANDOM_KINDS}
                  />
                </div>
                {row.kind === "choice" ? (
                  <div style={{ flex: "1 1 200px" }}>
                    <TextField
                      label="Opções (vírgula)"
                      value={row.options}
                      onChange={(v) => patchRow(i, { options: v })}
                      placeholder="resnet18, resnet50"
                      mono
                    />
                  </div>
                ) : (
                  <>
                    <div style={{ flex: "0 0 110px" }}>
                      <TextField
                        label="low"
                        value={row.low}
                        onChange={(v) => patchRow(i, { low: v })}
                        placeholder="0.001"
                        mono
                      />
                    </div>
                    <div style={{ flex: "0 0 110px" }}>
                      <TextField
                        label="high"
                        value={row.high}
                        onChange={(v) => patchRow(i, { high: v })}
                        placeholder="0.1"
                        mono
                      />
                    </div>
                  </>
                )}
              </>
            )}

            <button
              type="button"
              onClick={() => removeRow(i)}
              title="Remover parâmetro"
              disabled={rows.length === 1}
              style={{
                padding: "10px 12px",
                background: "transparent",
                border: "1px solid var(--vf-panel-stroke)",
                borderRadius: 8,
                color: "var(--vf-text-muted)",
                cursor: rows.length === 1 ? "not-allowed" : "pointer",
                opacity: rows.length === 1 ? 0.4 : 1,
              }}
            >
              ×
            </button>
          </div>
        ))}
      </div>

      <button
        type="button"
        onClick={addRow}
        style={{
          padding: "8px 14px",
          marginBottom: 16,
          background: "rgba(255,255,255,0.03)",
          border: "1px dashed var(--vf-panel-stroke)",
          borderRadius: 10,
          color: "var(--vf-text-dim)",
          fontFamily: "var(--font-mono)",
          fontSize: 11,
          cursor: "pointer",
        }}
      >
        + adicionar parâmetro
      </button>

      <div style={{ display: "flex", gap: 12, alignItems: "flex-end", flexWrap: "wrap" }}>
        <div style={{ minWidth: 180 }}>
          <SelectField
            label="Métrica de ranking"
            value={metric}
            onChange={setMetric}
            options={metrics}
          />
        </div>
        {mode === "random" && (
          <>
            <div style={{ width: 120 }}>
              <NumberField
                label="Nº de trials"
                value={nTrials}
                onChange={(v) => setNTrials(Math.max(1, Math.round(v)))}
                min={1}
                step={1}
              />
            </div>
            <div style={{ width: 110 }}>
              <NumberField
                label="Seed"
                value={seed}
                onChange={(v) => setSeed(Math.max(0, Math.round(v)))}
                min={0}
                step={1}
              />
            </div>
          </>
        )}
        <button
          type="button"
          onClick={() =>
            onSweep({ mode, search_space: searchSpace, metric, n_trials: nTrials, seed })
          }
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
          ⛓ Rodar sweep · {trialCount} trial{trialCount === 1 ? "" : "s"}
        </button>
      </div>
    </div>
  );
}
