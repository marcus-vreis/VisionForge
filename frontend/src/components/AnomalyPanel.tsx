import { useState } from "react";
import { fetchTaskSchema, pickDatasetFolder } from "../api/client";
import {
  ANOMALY_BACKBONES,
  ANOMALY_MODELS,
  anomalyFormFromPayload,
  buildAnomalyPayload,
  isPatchCore,
  type AnomalyForm,
} from "../lib/anomaly-models";
import { exportConfigToYaml, validateParsedConfig } from "../lib/yaml-config";
import type { ValidationError } from "../hooks/useExperiment";
import { NumberField, SelectField, Segmented, TextField, Toggle } from "./controls";
import { ExperimentHeader, type PanelStrategy } from "./ExperimentHeader";
import { ReplicatesCard } from "./ReplicatesCard";
import { AnomalyDatasetStats } from "./TaskDatasetStats";
import { SweepCard, type SweepPayload } from "./SweepCard";
import { TransformsSection } from "./TransformsSection";
import type { ReplicatesPayload } from "../lib/replicates-form";

const COMPARE_METRICS = [
  { value: "auroc", label: "AUROC" },
  { value: "image_f1", label: "F1 (imagem)" },
];

const SWEEP_PATH_HINTS = [
  "model.latent_dim",
  "model.coreset_ratio",
  "training.learning_rate",
];

interface AnomalyPanelProps {
  formData: AnomalyForm;
  setFormData: (updater: (prev: AnomalyForm) => AnomalyForm) => void;
  accent: string;
  validationErrors: ValidationError[];
  busy?: boolean;
  onSweep?: (payload: SweepPayload) => void;
  onReplicates?: (payload: ReplicatesPayload) => void;
  /** Lets App label and route the main Treinar button by the active strategy. */
  onStrategyChange?: (strategy: PanelStrategy) => void;
  /** Incremented by Treinar so the selected strategy's card runs. */
  runSignal?: number;
}

const sectionLabel: React.CSSProperties = {
  fontFamily: "var(--font-mono)",
  fontSize: 10,
  letterSpacing: "0.22em",
  textTransform: "uppercase",
  color: "var(--vf-text-muted)",
  marginBottom: 12,
};

const grid: React.CSSProperties = {
  display: "grid",
  gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
  gap: 14,
};

/** Schema-aligned form for an MVTec-style anomaly-detection run. */
export function AnomalyPanel({
  formData,
  setFormData,
  accent,
  validationErrors,
  busy,
  onSweep,
  onReplicates,
  onStrategyChange,
  runSignal,
}: AnomalyPanelProps) {
  const [picking, setPicking] = useState(false);
  const [strategy, setStrategy] = useState<PanelStrategy>("simple");

  const setModel = (patch: Partial<AnomalyForm["model"]>) =>
    setFormData((p) => ({ ...p, model: { ...p.model, ...patch } }));
  const setData = (patch: Partial<AnomalyForm["data"]>) =>
    setFormData((p) => ({ ...p, data: { ...p.data, ...patch } }));
  const setTraining = (patch: Partial<AnomalyForm["training"]>) =>
    setFormData((p) => ({ ...p, training: { ...p.training, ...patch } }));
  const setTransforms = (patch: Partial<AnomalyForm["transforms"]>) =>
    setFormData((p) => ({ ...p, transforms: { ...p.transforms, ...patch } }));
  const setPreprocessing = (steps: AnomalyForm["preprocessing"]) =>
    setFormData((p) => ({ ...p, preprocessing: steps }));

  const onPickFolder = async () => {
    setPicking(true);
    try {
      const res = await pickDatasetFolder();
      if (!res.cancelled && res.path) setData({ base_dir: res.path });
    } finally {
      setPicking(false);
    }
  };

  const patchcore = isPatchCore(formData);

  const card: React.CSSProperties = {
    background: "var(--vf-panel)",
    border: "1px solid var(--vf-panel-stroke)",
    borderRadius: 18,
    padding: 26,
    backdropFilter: "blur(14px)",
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
      {validationErrors.length > 0 && (
        <div
          style={{
            padding: "12px 16px",
            background: "oklch(0.704 0.191 22.216 / 0.10)",
            border: "1px solid oklch(0.704 0.191 22.216 / 0.4)",
            borderRadius: 12,
            fontFamily: "var(--font-mono)",
            fontSize: 12,
            color: "oklch(0.85 0.14 22)",
          }}
        >
          {validationErrors.slice(0, 5).map((e, i) => (
            <div key={i}>
              {e.field.join(" › ")}: {e.message}
            </div>
          ))}
        </div>
      )}

      {/* Cabeçalho canônico: nome + YAML + estratégia numa caixa (ADR-059) */}
      <ExperimentHeader
        name={formData.name}
        onNameChange={(v) => setFormData((p) => ({ ...p, name: v }))}
        placeholder="anomaly_001"
        strategy={strategy}
        onStrategyChange={(s) => {
          setStrategy(s);
          onStrategyChange?.(s);
        }}
        onExportYaml={() =>
          exportConfigToYaml(buildAnomalyPayload(formData), formData.name)
        }
        onImportConfig={async (data) => {
          try {
            const schema = await fetchTaskSchema("anomaly");
            const issues = validateParsedConfig(data, schema, schema.$defs ?? {});
            if (issues.length > 0) {
              return issues
                .slice(0, 5)
                .map((i) => `${i.field.join(" › ")}: ${i.message}`)
                .join("\n");
            }
          } catch {
            // schema unavailable → import tolerantly; o 422 do submit cobre.
          }
          setFormData(() => anomalyFormFromPayload(data));
          return null;
        }}
      />
      {strategy === "sweep" && onSweep && (
        <SweepCard
          metrics={COMPARE_METRICS}
          pathHints={SWEEP_PATH_HINTS}
          modelOptions={ANOMALY_MODELS}
          accent={accent}
          disabled={busy}
          onSweep={onSweep}
          runSignal={runSignal}
        />
      )}
      {strategy === "replicates" && onReplicates && (
        <ReplicatesCard
          metrics={COMPARE_METRICS}
          accent={accent}
          disabled={busy}
          onReplicates={onReplicates}
          runSignal={runSignal}
        />
      )}

      {/* Modelo */}
      <div style={card}>
        <div style={sectionLabel}>Modelo · anomalia</div>
        <div style={grid}>
          <Segmented
            label="Método"
            value={formData.model.name}
            onChange={(v) => setModel({ name: v })}
            options={ANOMALY_MODELS}
            hint="abordagem"
          />
          {patchcore ? (
            <>
              <SelectField
                label="Backbone"
                value={formData.model.backbone}
                onChange={(v) => setModel({ backbone: v })}
                options={ANOMALY_BACKBONES}
                hint="extrator congelado"
              />
              <NumberField
                label="Coreset ratio"
                value={formData.model.coreset_ratio}
                onChange={(v) => setModel({ coreset_ratio: v })}
                min={0.01}
                max={1}
                step={0.01}
                hint="subamostra do banco"
              />
              <Toggle
                label="Backbone pré-treinado"
                value={formData.model.pretrained}
                onChange={(v) => setModel({ pretrained: v })}
                hint="ImageNet"
              />
            </>
          ) : (
            <NumberField
              label="Latent dim"
              value={formData.model.latent_dim}
              onChange={(v) => setModel({ latent_dim: Math.round(v) })}
              min={1}
              step={1}
              hint="gargalo do autoencoder"
            />
          )}
        </div>
      </div>

      {/* Treinamento */}
      <div style={card}>
        <div style={sectionLabel}>Treinamento</div>
        <div style={grid}>
          <NumberField
            label="Épocas"
            value={formData.training.epochs}
            onChange={(v) => setTraining({ epochs: Math.round(v) })}
            min={1}
            step={1}
            hint={patchcore ? "ignorado no PatchCore" : undefined}
          />
          <NumberField
            label="Batch size"
            value={formData.training.batch_size}
            onChange={(v) => setTraining({ batch_size: Math.round(v) })}
            min={1}
            step={1}
            hint="qualquer inteiro"
          />
          <NumberField
            label="Learning rate"
            value={formData.training.learning_rate}
            onChange={(v) => setTraining({ learning_rate: v })}
            min={0.000001}
            step={0.0001}
          />
          <NumberField
            label="Threshold %ile"
            value={formData.training.threshold_percentile}
            onChange={(v) => setTraining({ threshold_percentile: v })}
            min={0}
            max={100}
            step={1}
            hint="corte sobre scores normais"
          />
          <Segmented
            label="Otimizador"
            value={formData.training.optimizer}
            onChange={(v) => setTraining({ optimizer: v })}
            options={[
              { value: "adam", label: "Adam" },
              { value: "sgd", label: "SGD" },
              { value: "adamw", label: "AdamW" },
            ]}
          />
          <NumberField
            label="Early stop"
            value={formData.training.early_stopping_patience}
            onChange={(v) => setTraining({ early_stopping_patience: Math.round(v) })}
            min={1}
            step={1}
            hint="paciência"
          />
          <NumberField
            label="Seed"
            value={formData.training.seed}
            onChange={(v) => setTraining({ seed: Math.round(v) })}
            min={0}
            step={1}
          />
          <Toggle
            label="Determinístico"
            value={formData.training.deterministic}
            onChange={(v) => setTraining({ deterministic: v })}
            hint="reprodutível, mais lento"
          />
        </div>
      </div>

      {/* Dataset */}
      <div style={card}>
        <div style={sectionLabel}>Dataset (MVTec · normal-only no treino)</div>
        <div style={grid}>
          <div
            style={{
              gridColumn: "1 / -1",
              display: "flex",
              gap: 10,
              alignItems: "flex-end",
            }}
          >
            <div style={{ flex: 1 }}>
              <TextField
                label="Pasta base"
                value={formData.data.base_dir}
                onChange={(v) => setData({ base_dir: v })}
                placeholder="…/categoria (train/good, test/good, test/<defeito>)"
                hint="raiz do dataset"
                mono
              />
            </div>
            <button
              type="button"
              onClick={() => void onPickFolder()}
              disabled={picking}
              style={{
                padding: "12px 16px",
                background: "var(--accent-soft)",
                border: `1px solid ${accent}`,
                borderRadius: 10,
                color: "var(--vf-text)",
                fontFamily: "var(--font-mono)",
                fontSize: 12,
                cursor: picking ? "default" : "pointer",
                whiteSpace: "nowrap",
              }}
            >
              {picking ? "…" : "📁 Escolher"}
            </button>
          </div>
          <TextField
            label="Split de treino"
            value={formData.data.train_dir}
            onChange={(v) => setData({ train_dir: v })}
            mono
          />
          <TextField
            label="Split de teste"
            value={formData.data.test_dir}
            onChange={(v) => setData({ test_dir: v })}
            mono
          />
          <TextField
            label="Subpasta normal"
            value={formData.data.normal_dir}
            onChange={(v) => setData({ normal_dir: v })}
            hint="label 0 (ex. good)"
            mono
          />
          <NumberField
            label="Image size"
            value={formData.data.image_size}
            onChange={(v) => setData({ image_size: Math.round(v) })}
            min={32}
            step={32}
            suffix="px"
          />
        </div>
        <AnomalyDatasetStats
          baseDir={formData.data.base_dir}
          trainDir={formData.data.train_dir}
          testDir={formData.data.test_dir}
          normalDir={formData.data.normal_dir}
        />
      </div>

      {/* Pré-processamento + augmentação (ADR-059 brick A) — flips/rotações
          importam para defeitos sensíveis a orientação; antes aplicavam-se
          silenciosamente. */}
      <TransformsSection
        baseDir={formData.data.base_dir}
        steps={formData.preprocessing}
        onStepsChange={setPreprocessing}
        transforms={formData.transforms}
        onTransformsChange={setTransforms}
        imageSize={formData.data.image_size}
      />

    </div>
  );
}
