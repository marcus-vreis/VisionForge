import { useState } from "react";
import { fetchTaskSchema, pickDatasetFolder } from "../api/client";
import {
  REGRESSION_LOSSES,
  REGRESSION_MODELS,
  buildRegressionPayload,
  parseTargetColumns,
  regressionFormFromPayload,
  type RegressionForm,
} from "../lib/regression-models";
import { exportConfigToYaml, validateParsedConfig } from "../lib/yaml-config";
import type { ValidationError } from "../hooks/useExperiment";
import { NumberField, SelectField, Segmented, TextField, Toggle } from "./controls";
import { ExperimentHeader, type PanelStrategy } from "./ExperimentHeader";
import { ReplicatesCard } from "./ReplicatesCard";
import { RegressionDatasetStats } from "./TaskDatasetStats";
import { SweepCard, type SweepPayload } from "./SweepCard";
import { TransformsSection } from "./TransformsSection";
import type { ReplicatesPayload } from "../lib/replicates-form";

const COMPARE_METRICS = [
  { value: "r2", label: "R²" },
  { value: "rmse", label: "RMSE" },
  { value: "mae", label: "MAE" },
  { value: "mse", label: "MSE" },
];

const SWEEP_PATH_HINTS = [
  "training.learning_rate",
  "training.batch_size",
  "model.name",
];

interface RegressionPanelProps {
  formData: RegressionForm;
  setFormData: (updater: (prev: RegressionForm) => RegressionForm) => void;
  accent: string;
  validationErrors: ValidationError[];
  busy?: boolean;
  onSweep?: (payload: SweepPayload) => void;
  onReplicates?: (payload: ReplicatesPayload) => void;
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

/** Schema-aligned form for a CSV-manifest image-regression run. */
export function RegressionPanel({
  formData,
  setFormData,
  accent,
  validationErrors,
  busy,
  onSweep,
  onReplicates,
}: RegressionPanelProps) {
  const [picking, setPicking] = useState(false);
  const [strategy, setStrategy] = useState<PanelStrategy>("simple");

  const setModel = (patch: Partial<RegressionForm["model"]>) =>
    setFormData((p) => ({ ...p, model: { ...p.model, ...patch } }));
  const setData = (patch: Partial<RegressionForm["data"]>) =>
    setFormData((p) => ({ ...p, data: { ...p.data, ...patch } }));
  const setTraining = (patch: Partial<RegressionForm["training"]>) =>
    setFormData((p) => ({ ...p, training: { ...p.training, ...patch } }));
  const setTransforms = (patch: Partial<RegressionForm["transforms"]>) =>
    setFormData((p) => ({ ...p, transforms: { ...p.transforms, ...patch } }));
  const setPreprocessing = (steps: RegressionForm["preprocessing"]) =>
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

  const numTargets = Math.max(parseTargetColumns(formData.data.target_columns).length, 1);

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
        placeholder="regression_001"
        strategy={strategy}
        onStrategyChange={setStrategy}
        onExportYaml={() =>
          exportConfigToYaml(buildRegressionPayload(formData), formData.name)
        }
        onImportConfig={async (data) => {
          try {
            const schema = await fetchTaskSchema("regression");
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
          setFormData(() => regressionFormFromPayload(data));
          return null;
        }}
      />
      {strategy === "sweep" && onSweep && (
        <SweepCard
          metrics={COMPARE_METRICS}
          pathHints={SWEEP_PATH_HINTS}
          modelOptions={REGRESSION_MODELS}
          accent={accent}
          disabled={busy}
          onSweep={onSweep}
        />
      )}
      {strategy === "replicates" && onReplicates && (
        <ReplicatesCard
          metrics={COMPARE_METRICS}
          accent={accent}
          disabled={busy}
          onReplicates={onReplicates}
        />
      )}

      {/* Modelo */}
      <div style={card}>
        <div style={sectionLabel}>Modelo · regressão</div>
        <div style={grid}>
          <SelectField
            label="Backbone"
            value={formData.model.name}
            onChange={(v) => setModel({ name: v })}
            options={REGRESSION_MODELS}
            hint="CNN"
          />
          <div>
            <div style={sectionLabel}>Nº de alvos</div>
            <div
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: 22,
                color: "var(--vf-text)",
                padding: "8px 0",
              }}
            >
              {numTargets}
              <span
                style={{
                  fontSize: 11,
                  color: "var(--vf-text-muted)",
                  marginLeft: 8,
                }}
              >
                derivado das colunas-alvo
              </span>
            </div>
          </div>
          <Toggle
            label="Pesos pré-treinados"
            value={formData.model.pretrained}
            onChange={(v) => setModel({ pretrained: v })}
            hint="ImageNet"
          />
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
          <Segmented
            label="Loss"
            value={formData.training.loss}
            onChange={(v) => setTraining({ loss: v })}
            options={REGRESSION_LOSSES}
            hint="critério"
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
        </div>
      </div>

      {/* Transfer learning */}
      <div style={card}>
        <div style={sectionLabel}>Transfer learning</div>
        <div style={grid}>
          <Segmented
            label="Modo"
            value={formData.transfer}
            onChange={(v) =>
              setFormData((p) => ({
                ...p,
                transfer: v as RegressionForm["transfer"],
              }))
            }
            options={[
              { value: "none", label: "Completo" },
              { value: "feature_extraction", label: "Feature extr." },
              { value: "fine_tuning", label: "Fine-tuning" },
            ]}
            hint="backbone compartilhado"
          />
          {formData.transfer === "fine_tuning" && (
            <NumberField
              label="Backbone LR ×"
              value={formData.backbone_lr_multiplier}
              onChange={(v) =>
                setFormData((p) => ({ ...p, backbone_lr_multiplier: v }))
              }
              min={0.0001}
              max={1}
              step={0.05}
              hint="LR do backbone = LR × isto"
            />
          )}
        </div>
      </div>

      {/* Dataset */}
      <div style={card}>
        <div style={sectionLabel}>Dataset (CSV manifest)</div>
        <div style={grid}>
          <div style={{ gridColumn: "1 / -1", display: "flex", gap: 10, alignItems: "flex-end" }}>
            <div style={{ flex: 1 }}>
              <TextField
                label="Pasta base"
                value={formData.data.base_dir}
                onChange={(v) => setData({ base_dir: v })}
                placeholder="…/dataset (train.csv, val.csv, images/)"
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
            label="Subpasta de imagens"
            value={formData.data.images_dir}
            onChange={(v) => setData({ images_dir: v })}
            hint="relativa à base"
            mono
          />
          <TextField
            label="Coluna da imagem"
            value={formData.data.image_column}
            onChange={(v) => setData({ image_column: v })}
            hint="cabeçalho do CSV"
            mono
          />
          <TextField
            label="Colunas-alvo"
            value={formData.data.target_columns}
            onChange={(v) => setData({ target_columns: v })}
            placeholder="target  ou  x,y,z"
            hint="separadas por vírgula"
            mono
          />
          <TextField
            label="CSV de treino"
            value={formData.data.train_csv}
            onChange={(v) => setData({ train_csv: v })}
            mono
          />
          <TextField
            label="CSV de validação"
            value={formData.data.val_csv}
            onChange={(v) => setData({ val_csv: v })}
            mono
          />
          <TextField
            label="CSV de teste"
            value={formData.data.test_csv}
            onChange={(v) => setData({ test_csv: v })}
            hint="opcional"
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
        <RegressionDatasetStats
          baseDir={formData.data.base_dir}
          imagesDir={formData.data.images_dir}
          trainCsv={formData.data.train_csv}
          valCsv={formData.data.val_csv}
          testCsv={formData.data.test_csv}
          imageColumn={formData.data.image_column}
          targetColumns={formData.data.target_columns}
        />
      </div>

      {/* Pré-processamento + augmentação (ADR-059 brick A) */}
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
