import { useRef, useState } from "react";
import { Segmented, TextField } from "./controls";
import { importConfigFromYaml } from "../lib/yaml-config";

/** Which experiment strategy the standalone-task panel is set to. The strategy
 *  cards (SweepCard / ReplicatesCard) render only for the selected mode. */
export type PanelStrategy = "simple" | "sweep" | "replicates";

interface ExperimentHeaderProps {
  name: string;
  onNameChange: (name: string) => void;
  placeholder: string;
  strategy: PanelStrategy;
  onStrategyChange: (value: PanelStrategy) => void;
  /** Serialize the current form and trigger the .yaml download. */
  onExportYaml: () => void;
  /** Apply a parsed YAML config to the form; return an error message to show,
   *  or null on success. */
  onImportConfig: (
    data: Record<string, unknown>,
  ) => string | null | Promise<string | null>;
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

const yamlBtnStyle: React.CSSProperties = {
  padding: "8px 14px",
  background: "var(--accent-soft)",
  border: "1px solid var(--accent-vf)",
  borderRadius: 10,
  color: "var(--vf-text)",
  fontFamily: "var(--font-mono)",
  fontSize: 11,
  letterSpacing: "0.10em",
  textTransform: "uppercase",
  cursor: "pointer",
  whiteSpace: "nowrap",
  lineHeight: 1,
};

const yamlBtnSecondaryStyle: React.CSSProperties = {
  ...yamlBtnStyle,
  background: "transparent",
  border: "1px solid var(--vf-panel-stroke)",
  color: "var(--vf-text-dim)",
};

const STRATEGY_HINTS: Record<PanelStrategy, string> = {
  simple: "um treino com a config abaixo (botão Treinar)",
  sweep: "grid / random / optuna sobre a config abaixo",
  replicates: "mesma config, N seeds → média ± IC 95%",
};

/** Canonical experiment header (ADR-059): every task panel opens with the same
 *  card — experiment name + YAML export/import side by side, and the strategy
 *  selector below, in the same box — mirroring the classification layout. */
export function ExperimentHeader({
  name,
  onNameChange,
  placeholder,
  strategy,
  onStrategyChange,
  onExportYaml,
  onImportConfig,
}: ExperimentHeaderProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [importError, setImportError] = useState<string | null>(null);
  const [importOk, setImportOk] = useState<string | null>(null);

  const handleFile = async (file: File) => {
    setImportError(null);
    setImportOk(null);
    const parsed = await importConfigFromYaml(file);
    if ("error" in parsed) {
      setImportError(parsed.error);
      return;
    }
    const problem = await onImportConfig(parsed.data);
    if (problem) {
      setImportError(problem);
    } else {
      setImportOk(`✓ ${file.name} importado`);
    }
  };

  return (
    <div style={card}>
      <input
        ref={fileInputRef}
        type="file"
        accept=".yaml,.yml"
        style={{ display: "none" }}
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) void handleFile(file);
          e.target.value = "";
        }}
      />
      <div style={sectionLabel}>Experimento</div>
      <div
        style={{
          display: "flex",
          gap: 10,
          alignItems: "flex-end",
          flexWrap: "wrap",
        }}
      >
        <div style={{ flex: "1 1 280px", maxWidth: 420 }}>
          <TextField
            label="Nome do experimento"
            value={name}
            onChange={onNameChange}
            placeholder={placeholder}
            hint="usado na pasta de saída e no histórico"
            mono
          />
        </div>
        <div style={{ display: "flex", gap: 8, paddingBottom: 2 }}>
          <button
            type="button"
            onClick={onExportYaml}
            style={yamlBtnStyle}
            title="Exportar configuração atual como arquivo .yaml"
          >
            ↓ Exportar YAML
          </button>
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            style={yamlBtnSecondaryStyle}
            title="Importar configuração a partir de um arquivo .yaml"
          >
            ↑ Importar YAML
          </button>
        </div>
      </div>

      {importError && (
        <div
          style={{
            marginTop: 12,
            padding: "10px 14px",
            background: "oklch(0.704 0.191 22.216 / 0.10)",
            border: "1px solid oklch(0.704 0.191 22.216 / 0.4)",
            borderRadius: 10,
            fontFamily: "var(--font-mono)",
            fontSize: 11.5,
            whiteSpace: "pre-line",
            color: "oklch(0.85 0.14 22)",
          }}
        >
          {importError}
        </div>
      )}
      {importOk && !importError && (
        <div
          style={{
            marginTop: 12,
            fontFamily: "var(--font-mono)",
            fontSize: 11.5,
            color: "oklch(0.85 0.16 150)",
          }}
        >
          {importOk}
        </div>
      )}

      <div
        style={{
          marginTop: 18,
          paddingTop: 18,
          borderTop: "1px solid var(--vf-panel-stroke)",
        }}
      >
        <div style={sectionLabel}>Estratégia de experimento</div>
        <div style={{ maxWidth: 460 }}>
          <Segmented
            label="Modo"
            value={strategy}
            onChange={(v) => onStrategyChange(v as PanelStrategy)}
            options={[
              { value: "simple", label: "Treino simples" },
              { value: "sweep", label: "Sweep" },
              { value: "replicates", label: "Réplicas" },
            ]}
            hint={STRATEGY_HINTS[strategy]}
          />
        </div>
      </div>
    </div>
  );
}
