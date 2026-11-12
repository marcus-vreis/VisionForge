import { useEffect, useRef, useState } from "react";
import { RunSummary } from "./RunSummary";
import { useT } from "../i18n/useT";
import { NumberField, SelectField, Segmented, TextField } from "./controls";
import { parseSeeds, type ReplicatesSeedMode } from "../lib/replicates-form";
import {
  COMPARISON_ALPHA,
  MAX_COMPARISON_SEEDS,
  buildReplicatedComparisonPayload,
  comparisonSeedCount,
  comparisonVariantCount,
  powerFloor,
  type ReplicatedComparisonPayload,
} from "../lib/replicated-comparison-form";

interface Option {
  value: string;
  label: string;
}

interface ReplicatedComparisonCardProps {
  /** Metrics the tests run on; the first is the default (the task's primary). */
  metrics: Option[];
  /** Suggested dot-paths shown under the field (e.g. training.learning_rate). */
  pathHints: string[];
  accent: string;
  disabled?: boolean;
  onReplicatedComparison: (payload: ReplicatedComparisonPayload) => void;
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

const noteText: React.CSSProperties = {
  margin: 0,
  fontFamily: "var(--font-mono)",
  fontSize: 11.5,
  lineHeight: 1.6,
};

// RunSummary's warning colour, so the card speaks with one voice.
const WARN_COLOR = "oklch(calc(0.85 - var(--lt-flip, 0) * 0.4) 0.14 70)";

/** What the comparison costs and, when the seeds are too few for the rank
 *  test to ever reject, why its verdict could not be "significant". */
export function ComparisonCost({ variants, seeds }: { variants: number; seeds: number }) {
  const t = useT();
  const floor = powerFloor(seeds);
  return (
    <div style={{ display: "grid", gap: 8, marginTop: 16 }}>
      <p style={{ ...noteText, color: "var(--vf-text-dim)" }}>
        {t.replicatedComparisonCard.cost(variants, seeds, variants * seeds)}
      </p>
      {floor !== null && (
        <p role="note" style={{ ...noteText, color: WARN_COLOR }}>
          ⚠ {t.replicatedComparisonCard.power(seeds, String(floor), String(COMPARISON_ALPHA))}
        </p>
      )}
    </div>
  );
}

/** Paired comparison launcher (ADR-061, ADR-193): the form is the baseline,
 *  one field takes 1 to 3 other values, and every variant trains on the same
 *  seeds so the backend can test each pair (Holm-corrected). */
export function ReplicatedComparisonCard({
  metrics,
  pathHints,
  accent,
  disabled,
  onReplicatedComparison,
  runSignal,
}: ReplicatedComparisonCardProps) {
  const t = useT();
  const c = t.replicatedComparisonCard;
  const [path, setPath] = useState("");
  const [rawValues, setRawValues] = useState("");
  const [seedMode, setSeedMode] = useState<ReplicatesSeedMode>("auto");
  const [nReplicates, setNReplicates] = useState(5);
  const [rawSeeds, setRawSeeds] = useState("42, 43, 44, 45, 46");
  const [metric, setMetric] = useState(metrics[0]?.value ?? "");

  const fields = { path, rawValues, seedMode, nReplicates, rawSeeds, metric };
  const built = buildReplicatedComparisonPayload(t, fields);
  const problem = "problem" in built ? built.problem : null;
  const variants = comparisonVariantCount(rawValues);
  const seeds = comparisonSeedCount(fields);
  const canRun = !disabled && problem === null;

  const run = () => {
    if ("payload" in built) onReplicatedComparison(built.payload);
  };

  // Fire on a new signal only — the card keeps owning its own fields, so the
  // effect deliberately does not depend on them (it would re-fire on typing).
  // Set when the bottom bar asked to run and this card could not.
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
      <div style={sectionLabel}>{c.title}</div>
      <p
        style={{
          margin: "0 0 14px",
          fontFamily: "var(--font-mono)",
          fontSize: 11,
          lineHeight: 1.6,
          color: "var(--vf-text-muted)",
        }}
      >
        {c.description(pathHints.join(", "))}
      </p>

      <div style={{ display: "flex", gap: 12, alignItems: "flex-end", flexWrap: "wrap" }}>
        <div style={{ flex: "1 1 220px" }}>
          <TextField
            label={c.parameter}
            value={path}
            onChange={setPath}
            placeholder={pathHints[0]}
            mono
          />
        </div>
        <div style={{ flex: "1 1 220px" }}>
          <TextField
            label={c.values}
            value={rawValues}
            onChange={setRawValues}
            placeholder={c.valuesPlaceholder}
            hint={c.valuesHint}
            mono
          />
        </div>
      </div>

      <div
        style={{
          display: "flex",
          gap: 12,
          alignItems: "flex-end",
          flexWrap: "wrap",
          marginTop: 16,
        }}
      >
        <div style={{ minWidth: 220 }}>
          <Segmented
            label={t.replicatesCard.seeds}
            value={seedMode}
            onChange={(v) => setSeedMode(v as ReplicatesSeedMode)}
            options={[
              { value: "auto", label: t.replicatesCard.automatic },
              { value: "explicit", label: t.replicatesCard.explicit },
            ]}
            hint={
              seedMode === "auto" ? t.replicatesCard.automaticHint : t.replicatesCard.explicitHint
            }
          />
        </div>
        {seedMode === "auto" ? (
          <div style={{ width: 150 }}>
            <NumberField
              label={c.count}
              value={nReplicates}
              onChange={(v) =>
                setNReplicates(Math.min(MAX_COMPARISON_SEEDS, Math.max(2, Math.round(v))))
              }
              min={2}
              max={MAX_COMPARISON_SEEDS}
              step={1}
            />
          </div>
        ) : (
          <div style={{ flex: "1 1 240px" }}>
            <TextField
              label={t.replicatesCard.seedList}
              value={rawSeeds}
              onChange={setRawSeeds}
              placeholder="42, 43, 44, 45, 46"
              hint={t.replicatesCard.seedCount(parseSeeds(rawSeeds).length)}
              mono
            />
          </div>
        )}
        <div style={{ minWidth: 170 }}>
          <SelectField
            label={t.replicatesCard.headlineMetric}
            value={metric}
            onChange={setMetric}
            options={metrics}
          />
        </div>
        <RunSummary
          text={c.summary(variants * seeds)}
          ready={canRun}
          reason={disabled ? null : problem}
          attempted={attempted}
          accent={accent}
        />
      </div>

      <ComparisonCost variants={variants} seeds={seeds} />
    </div>
  );
}
