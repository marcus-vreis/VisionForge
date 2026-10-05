import { paramHelp } from "../lib/param-help";
import { useT } from "../i18n/useT";
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
import { AdvancedFields } from "./AdvancedFields";
import { ExperimentHeader, type PanelStrategy } from "./ExperimentHeader";
import { ReplicatesCard } from "./ReplicatesCard";
import { AnomalyDatasetStats } from "./TaskDatasetStats";
import { SweepCard, type SweepPayload } from "./SweepCard";
import { TransformsSection } from "./TransformsSection";
import type { ReplicatesPayload } from "../lib/replicates-form";

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
  const t = useT();
  const compareMetrics = [
    { value: "auroc", label: "AUROC" },
    { value: "image_f1", label: t.anomalyPanel.imageF1 },
  ];
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
        placeholder={t.anomalyPanel.namePlaceholder}
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
          metrics={compareMetrics}
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
          metrics={compareMetrics}
          accent={accent}
          disabled={busy}
          onReplicates={onReplicates}
          runSignal={runSignal}
        />
      )}

      {/* Modelo */}
      <div style={card}>
        <div style={sectionLabel}>{t.anomalyPanel.model.title}</div>
        <div style={grid}>
          <Segmented
            label={t.anomalyPanel.model.method}
            value={formData.model.name}
            onChange={(v) => setModel({ name: v })}
            options={ANOMALY_MODELS}
            hint={t.anomalyPanel.model.methodHint}
          />
          {patchcore ? (
            <>
              <SelectField
                label={t.taskPanel.backbone}
                value={formData.model.backbone}
                onChange={(v) => setModel({ backbone: v })}
                options={ANOMALY_BACKBONES}
                hint={t.anomalyPanel.model.backboneHint}
              />
              <NumberField
                label={t.anomalyPanel.model.coresetRatio}
                value={formData.model.coreset_ratio}
                onChange={(v) => setModel({ coreset_ratio: v })}
                min={0.01}
                max={1}
                step={0.01}
                hint={t.anomalyPanel.model.coresetRatioHint}
                help={paramHelp(t, "coreset_ratio")}
              />
              <Toggle
                label={t.anomalyPanel.model.pretrainedBackbone}
                value={formData.model.pretrained}
                onChange={(v) => setModel({ pretrained: v })}
                hint={t.taskPanel.imageNet}
              />
            </>
          ) : (
            <NumberField
              label={t.anomalyPanel.model.latentDim}
              value={formData.model.latent_dim}
              onChange={(v) => setModel({ latent_dim: Math.round(v) })}
              min={1}
              step={1}
              hint={t.anomalyPanel.model.latentDimHint}
            />
          )}
        </div>
      </div>

      {/* Treinamento */}
      <div style={card}>
        <div style={sectionLabel}>{t.taskPanel.training.title}</div>
        <div style={grid}>
          <NumberField
            label={t.taskPanel.training.epochs}
            value={formData.training.epochs}
            onChange={(v) => setTraining({ epochs: Math.round(v) })}
            min={1}
            step={1}
            hint={patchcore ? t.anomalyPanel.epochsHint : undefined}
            help={paramHelp(t, "epochs")}
          />
          <NumberField
            label={t.taskPanel.training.batchSize}
            value={formData.training.batch_size}
            onChange={(v) => setTraining({ batch_size: Math.round(v) })}
            min={1}
            step={1}
            hint={t.taskPanel.training.batchSizeHint}
            help={paramHelp(t, "batch_size")}
          />
          <NumberField
            label={t.taskPanel.training.learningRate}
            value={formData.training.learning_rate}
            onChange={(v) => setTraining({ learning_rate: v })}
            min={0.000001}
            step={0.0001}
            help={paramHelp(t, "learning_rate")}
          />
          <NumberField
            label={t.taskPanel.training.seed}
            value={formData.training.seed}
            onChange={(v) => setTraining({ seed: Math.round(v) })}
            min={0}
            step={1}
            help={paramHelp(t, "seed")}
          />
        </div>
        <AdvancedFields count={4}>
            <NumberField
            label={t.anomalyPanel.threshold}
            value={formData.training.threshold_percentile}
            onChange={(v) => setTraining({ threshold_percentile: v })}
            min={0}
            max={100}
            step={1}
            hint={t.anomalyPanel.thresholdHint}
          />
            <Segmented
            label={t.taskPanel.training.optimizer}
            value={formData.training.optimizer}
            onChange={(v) => setTraining({ optimizer: v })}
            options={[
              { value: "adam", label: "Adam" },
              { value: "sgd", label: "SGD" },
              { value: "adamw", label: "AdamW" },
            ]}
            help={paramHelp(t, "optimizer")}
          />
            <NumberField
            label={t.taskPanel.training.earlyStop}
            value={formData.training.early_stopping_patience}
            onChange={(v) => setTraining({ early_stopping_patience: Math.round(v) })}
            min={0}
            step={1}
            hint={t.taskPanel.training.earlyStopHint}
            help={paramHelp(t, "early_stopping_patience")}
            emptyValue={0}
          />
            <Toggle
            label={t.taskPanel.training.deterministic}
            value={formData.training.deterministic}
            onChange={(v) => setTraining({ deterministic: v })}
            hint={t.taskPanel.training.deterministicHint}
            help={paramHelp(t, "deterministic")}
          />
        </AdvancedFields>
      </div>

      {/* Dataset */}
      <div style={card}>
        <div style={sectionLabel}>{t.anomalyPanel.dataset.title}</div>
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
                label={t.taskPanel.dataset.baseDir}
                value={formData.data.base_dir}
                onChange={(v) => setData({ base_dir: v })}
                placeholder={t.anomalyPanel.dataset.baseDirPlaceholder}
                hint={t.taskPanel.dataset.baseDirHint}
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
              {picking ? "…" : t.taskPanel.dataset.browse}
            </button>
          </div>
          <TextField
            label={t.taskPanel.dataset.trainSplit}
            value={formData.data.train_dir}
            onChange={(v) => setData({ train_dir: v })}
            mono
          />
          <TextField
            label={t.taskPanel.dataset.testSplit}
            value={formData.data.test_dir}
            onChange={(v) => setData({ test_dir: v })}
            mono
          />
          <TextField
            label={t.anomalyPanel.dataset.normalDir}
            value={formData.data.normal_dir}
            onChange={(v) => setData({ normal_dir: v })}
            hint={t.anomalyPanel.dataset.normalDirHint}
            mono
          />
          <NumberField
            label={t.taskPanel.dataset.imageSize}
            value={formData.data.image_size}
            onChange={(v) => setData({ image_size: Math.round(v) })}
            min={32}
            step={32}
            suffix="px"
            help={paramHelp(t, "image_size")}
          />
        </div>
        <AnomalyDatasetStats
          baseDir={formData.data.base_dir}
          trainDir={formData.data.train_dir}
          testDir={formData.data.test_dir}
          normalDir={formData.data.normal_dir}
        />
      </div>

      {/* Pré-processamento + augmentação (ADR-059) — flips/rotações
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
