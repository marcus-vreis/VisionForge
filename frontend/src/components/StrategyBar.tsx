import { Segmented } from "./controls";

/** Which experiment strategy the standalone-task panel is set to. The strategy
 *  cards (SweepCard / ReplicatesCard) render only for the selected mode. */
export type PanelStrategy = "simple" | "sweep" | "replicates";

interface StrategyBarProps {
  value: PanelStrategy;
  onChange: (value: PanelStrategy) => void;
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

const HINTS: Record<PanelStrategy, string> = {
  simple: "um treino com a config abaixo (botão Treinar)",
  sweep: "grid / random / optuna sobre a config abaixo",
  replicates: "mesma config, N seeds → média ± IC 95%",
};

/** Canonical strategy selector for the standalone task panels (ADR-059 brick C)
 *  — mirrors classification's BlockSelector: one segmented control that decides
 *  which strategy surface renders, instead of stacked always-visible cards. */
export function StrategyBar({ value, onChange }: StrategyBarProps) {
  return (
    <div style={card}>
      <div style={sectionLabel}>Estratégia de experimento</div>
      <div style={{ maxWidth: 460 }}>
        <Segmented
          label="Modo"
          value={value}
          onChange={(v) => onChange(v as PanelStrategy)}
          options={[
            { value: "simple", label: "Treino simples" },
            { value: "sweep", label: "Sweep" },
            { value: "replicates", label: "Réplicas" },
          ]}
          hint={HINTS[value]}
        />
      </div>
    </div>
  );
}
