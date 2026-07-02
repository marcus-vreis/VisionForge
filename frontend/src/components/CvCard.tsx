import { useState } from "react";
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
export function CvCard({ accent, disabled, onCv }: CvCardProps) {
  const [nFolds, setNFolds] = useState(5);
  const [shuffle, setShuffle] = useState(true);
  const [foldSeed, setFoldSeed] = useState(42);

  const canRun = !disabled && nFolds >= 2;

  return (
    <div style={card}>
      <div style={sectionLabel}>Cross-validation (K-fold)</div>
      <p
        style={{
          margin: "0 0 14px",
          fontFamily: "var(--font-mono)",
          fontSize: 11,
          lineHeight: 1.6,
          color: "var(--vf-text-muted)",
        }}
      >
        Divide as linhas de treino em K folds: cada fold treina um modelo novo
        em K-1 partes e avalia na parte restante (nunca augmentada). O split de
        teste não é usado.
      </p>
      <div style={{ display: "flex", gap: 12, alignItems: "flex-end", flexWrap: "wrap" }}>
        <div style={{ width: 120 }}>
          <NumberField
            label="Nº de folds"
            value={nFolds}
            onChange={(v) => setNFolds(Math.min(20, Math.max(2, Math.round(v))))}
            min={2}
            max={20}
            step={1}
          />
        </div>
        <Toggle label="Shuffle" value={shuffle} onChange={setShuffle} hint="antes do split" />
        <div style={{ width: 120 }}>
          <NumberField
            label="Seed do split"
            value={foldSeed}
            onChange={(v) => setFoldSeed(Math.max(0, Math.round(v)))}
            min={0}
            step={1}
          />
        </div>
        <button
          type="button"
          onClick={() => onCv({ n_folds: nFolds, shuffle, fold_seed: foldSeed })}
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
          ⛓ Rodar CV · {nFolds} folds
        </button>
      </div>
    </div>
  );
}
