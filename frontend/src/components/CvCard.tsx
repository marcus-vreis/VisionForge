import { useEffect, useRef, useState } from "react";
import { useT } from "../i18n/useT";
import { NumberField, Toggle } from "./controls";

export interface CvPayload {
  n_folds: number;
  shuffle: boolean;
  fold_seed: number;
}

interface CvCardProps {
  accent: string;
  disabled?: boolean;
  onCv: (payload: CvPayload) => void;
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

/** K-fold cross-validation launcher for a standalone task (ADR-050): splits
 *  the pooled training rows into K folds and reports fold-a-fold metrics +
 *  mean ± std — the honest estimate when there is no big held-out val set. */
export function CvCard({ accent, disabled, onCv, runSignal }: CvCardProps) {
  const t = useT();
  const [nFolds, setNFolds] = useState(5);
  const [shuffle, setShuffle] = useState(true);
  const [foldSeed, setFoldSeed] = useState(42);

  const canRun = !disabled && nFolds >= 2;

  const run = () => onCv({ n_folds: nFolds, shuffle, fold_seed: foldSeed });

  // Fire on a new signal only — see ReplicatesCard for the rationale.
  const lastSignal = useRef(runSignal ?? 0);
  useEffect(() => {
    if (runSignal === undefined || runSignal === lastSignal.current) return;
    lastSignal.current = runSignal;
    if (canRun) run();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runSignal]);

  return (
    <div style={card}>
      <div style={sectionLabel}>{t.cvCard.title}</div>
      <p
        style={{
          margin: "0 0 14px",
          fontFamily: "var(--font-mono)",
          fontSize: 11,
          lineHeight: 1.6,
          color: "var(--vf-text-muted)",
        }}
      >
        {t.cvCard.description}
      </p>
      <div style={{ display: "flex", gap: 12, alignItems: "flex-end", flexWrap: "wrap" }}>
        <div style={{ width: 120 }}>
          <NumberField
            label={t.cvCard.folds}
            value={nFolds}
            onChange={(v) => setNFolds(Math.min(20, Math.max(2, Math.round(v))))}
            min={2}
            max={20}
            step={1}
          />
        </div>
        <Toggle label={t.cvCard.shuffle} value={shuffle} onChange={setShuffle} hint={t.cvCard.shuffleHint} />
        <div style={{ width: 120 }}>
          <NumberField
            label={t.cvCard.foldSeed}
            value={foldSeed}
            onChange={(v) => setFoldSeed(Math.max(0, Math.round(v)))}
            min={0}
            step={1}
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
          {t.cvCard.run(nFolds)}
        </button>
      </div>
    </div>
  );
}
