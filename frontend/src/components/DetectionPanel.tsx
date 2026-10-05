import { paramHelp } from "../lib/param-help";
import { useT } from "../i18n/useT";
import { useState } from "react";
import {
  fetchTaskSchema,
  pickDatasetFolder,
  pickDetectionYaml,
} from "../api/client";
import {
  DETECTION_BACKENDS,
  DETECTION_MODELS,
  DETECTION_OPTIMIZERS,
  buildDetectionDataPayload,
  buildDetectionTrainingPayload,
  defaultModelForBackend,
  detectionFormFromPayload,
  isValidModelForBackend,
  type DetectionAugmentationForm,
  type DetectionBackend,
  type DetectionForm,
  type DetectionOptimizer,
} from "../lib/detection-models";
import { exportConfigToYaml, validateParsedConfig } from "../lib/yaml-config";
import type { ValidationError } from "../hooks/useExperiment";
import {
  NumberField,
  SelectField,
  Segmented,
  TextField,
  Toggle,
  WorkersField,
} from "./controls";
import { AdvancedFields } from "./AdvancedFields";
import { DetectionDatasetStats } from "./DetectionDatasetStats";
import { PreprocessingPanel } from "./PreprocessingPanel";
import { ExperimentHeader, type PanelStrategy } from "./ExperimentHeader";
import { Rich } from "./Rich";
import { ReplicatesCard } from "./ReplicatesCard";
import { SweepCard, type SweepPayload } from "./SweepCard";
import type { ReplicatesPayload } from "../lib/replicates-form";

const COMPARE_METRICS = [
  { value: "map50_95", label: "mAP@50-95" },
  { value: "map50", label: "mAP@50" },
];

const SWEEP_PATH_HINTS = [
  "training.learning_rate",
  "training.epochs",
  "model.name",
];

interface DetectionPanelProps {
  formData: DetectionForm;
  setFormData: (updater: (prev: DetectionForm) => DetectionForm) => void;
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

/** Schema-aligned form for an Ultralytics/torchvision detection run. */
export function DetectionPanel({
  formData,
  setFormData,
  accent,
  validationErrors,
  busy,
  onSweep,
  onReplicates,
  onStrategyChange,
  runSignal,
}: DetectionPanelProps) {
  const t = useT();
  const [picking, setPicking] = useState(false);
  const [strategy, setStrategy] = useState<PanelStrategy>("simple");

  const setModel = (patch: Partial<DetectionForm["model"]>) =>
    setFormData((p) => ({ ...p, model: { ...p.model, ...patch } }));
  const setData = (patch: Partial<DetectionForm["data"]>) =>
    setFormData((p) => ({ ...p, data: { ...p.data, ...patch } }));
  const setTraining = (patch: Partial<DetectionForm["training"]>) =>
    setFormData((p) => ({ ...p, training: { ...p.training, ...patch } }));
  const setAug = (patch: Partial<DetectionAugmentationForm>) =>
    setFormData((p) => ({
      ...p,
      training: {
        ...p.training,
        augmentation: { ...p.training.augmentation, ...patch },
      },
    }));

  const isUltralytics = formData.model.backend === "ultralytics";

  const onBackendChange = (raw: string) => {
    const backend = raw as DetectionBackend;
    setFormData((p) => {
      const name = isValidModelForBackend(backend, p.model.name)
        ? p.model.name
        : defaultModelForBackend(backend);
      return { ...p, model: { ...p.model, backend, name } };
    });
  };

  const onPickFolder = async () => {
    setPicking(true);
    try {
      const res = await pickDatasetFolder();
      if (!res.cancelled && res.path) setData({ base_dir: res.path });
    } finally {
      setPicking(false);
    }
  };

  const onPickYaml = async () => {
    setPicking(true);
    try {
      const res = await pickDetectionYaml();
      if (!res.cancelled && res.path) setData({ data_yaml: res.path });
    } finally {
      setPicking(false);
    }
  };

  const pickButton: React.CSSProperties = {
    padding: "12px 16px",
    background: "var(--accent-soft)",
    border: `1px solid ${accent}`,
    borderRadius: 10,
    color: "var(--vf-text)",
    fontFamily: "var(--font-mono)",
    fontSize: 12,
    cursor: picking ? "default" : "pointer",
    whiteSpace: "nowrap",
  };

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
        placeholder={t.detectionPanel.namePlaceholder}
        strategy={strategy}
        onStrategyChange={(s) => {
          setStrategy(s);
          onStrategyChange?.(s);
        }}
        onExportYaml={() =>
          exportConfigToYaml(
            {
              name: formData.name,
              model: { ...formData.model },
              data: buildDetectionDataPayload(formData.data),
              training: buildDetectionTrainingPayload(formData.training),
            },
            formData.name,
          )
        }
        onImportConfig={async (data) => {
          try {
            const schema = await fetchTaskSchema("detection");
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
          setFormData(() => detectionFormFromPayload(data));
          return null;
        }}
      />
      {strategy === "sweep" && onSweep && (
        <SweepCard
          metrics={COMPARE_METRICS}
          pathHints={SWEEP_PATH_HINTS}
          modelOptions={DETECTION_MODELS[formData.model.backend]}
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
        <div style={sectionLabel}>{t.detectionPanel.model.title}</div>
        <div style={grid}>
          <Segmented
            label={t.detectionPanel.model.backend}
            value={formData.model.backend}
            onChange={onBackendChange}
            options={DETECTION_BACKENDS.map((b) => ({
              value: b,
              label: b === "ultralytics" ? "Ultralytics" : "Torchvision",
            }))}
            hint={t.detectionPanel.model.backendHint}
          />
          <SelectField
            label={t.detectionPanel.model.architecture}
            value={formData.model.name}
            onChange={(v) => setModel({ name: v })}
            options={DETECTION_MODELS[formData.model.backend]}
            hint={t.detectionPanel.model.architectureHint}
          />
          {/* Nº de classes vive na seção Dataset, ao lado de onde o dataset é
              escolhido — é dele que o número sai. Classificação já fazia isso;
              detecção pedia o número aqui e só confirmava lá embaixo. */}
          <Toggle
            label={t.detectionPanel.model.pretrained}
            value={formData.model.pretrained}
            onChange={(v) => setModel({ pretrained: v })}
            hint={t.detectionPanel.model.pretrainedHint}
          />
        </div>
      </div>

      {/* Treinamento */}
      <div style={card}>
        <div style={sectionLabel}>{t.detectionPanel.training.title}</div>
        <div style={grid}>
          <NumberField
            label={t.detectionPanel.training.epochs}
            value={formData.training.epochs}
            onChange={(v) => setTraining({ epochs: Math.round(v) })}
            min={1}
            step={1}
            help={paramHelp(t, "epochs")}
          />
          <NumberField
            label={t.detectionPanel.training.batchSize}
            value={formData.training.batch_size}
            onChange={(v) => setTraining({ batch_size: Math.round(v) })}
            min={1}
            step={1}
            hint={t.detectionPanel.training.batchSizeHint}
            help={paramHelp(t, "batch_size")}
          />
          <NumberField
            label={t.detectionPanel.training.learningRate}
            value={formData.training.learning_rate}
            onChange={(v) => setTraining({ learning_rate: v })}
            min={0.000001}
            step={0.001}
            hint={t.detectionPanel.training.learningRateHint}
            help={paramHelp(t, "learning_rate")}
          />
          <NumberField
            label={t.detectionPanel.training.seed}
            value={formData.training.seed}
            onChange={(v) => setTraining({ seed: Math.round(v) })}
            min={0}
            step={1}
            help={paramHelp(t, "seed")}
          />
        </div>
        <AdvancedFields count={6}>
            <NumberField
            label={t.detectionPanel.training.patience}
            value={formData.training.patience}
            onChange={(v) => setTraining({ patience: Math.round(v) })}
            min={0}
            step={1}
            hint={t.detectionPanel.training.patienceHint}
            help={paramHelp(t, "patience")}
            emptyValue={0}
          />
            <Toggle
            label={t.detectionPanel.training.deterministic}
            value={formData.training.deterministic}
            onChange={(v) => setTraining({ deterministic: v })}
            hint={t.detectionPanel.training.deterministicHint}
            help={paramHelp(t, "deterministic")}
          />
            <WorkersField
            value={formData.training.workers}
            onChange={(v) => setTraining({ workers: v })}
          />
            <SelectField
            label={t.detectionPanel.training.optimizer}
            value={formData.training.optimizer}
            onChange={(v) =>
              setTraining({ optimizer: v as DetectionOptimizer })
            }
            options={DETECTION_OPTIMIZERS.map((o) => ({ value: o, label: o }))}
            hint={t.detectionPanel.training.optimizerHint}
            help={paramHelp(t, "optimizer")}
          />
            <NumberField
            label={t.detectionPanel.training.momentum}
            value={formData.training.momentum}
            onChange={(v) => setTraining({ momentum: v })}
            min={0}
            max={1}
            step={0.001}
            hint={t.detectionPanel.training.momentumHint}
            help={paramHelp(t, "momentum")}
          />
            <NumberField
            label={t.detectionPanel.training.weightDecay}
            value={formData.training.weight_decay}
            onChange={(v) => setTraining({ weight_decay: v })}
            min={0}
            step={0.0001}
            hint={t.detectionPanel.training.weightDecayHint}
            help={paramHelp(t, "weight_decay")}
            emptyValue={0}
          />
        </AdvancedFields>
      </div>

      {isUltralytics && (
        <>
          {/* Schedule & loss */}
          <div style={card}>
            <div style={sectionLabel}>{t.detectionPanel.schedule.title}</div>
            <div style={grid}>
              <NumberField
                label={t.detectionPanel.schedule.lrf}
                value={formData.training.lrf}
                onChange={(v) => setTraining({ lrf: v })}
                min={0.000001}
                step={0.001}
                hint={t.detectionPanel.schedule.lrfHint}
                help={paramHelp(t, "lrf")}
              />
              <Toggle
                label={t.detectionPanel.schedule.cosLr}
                value={formData.training.cos_lr}
                onChange={(v) => setTraining({ cos_lr: v })}
                hint={t.detectionPanel.schedule.cosLrHint}
                help={paramHelp(t, "cos_lr")}
              />
              <NumberField
                label={t.detectionPanel.schedule.warmupEpochs}
                value={formData.training.warmup_epochs}
                onChange={(v) => setTraining({ warmup_epochs: v })}
                min={0}
                step={0.5}
                help={paramHelp(t, "warmup_epochs")}
              />
              <NumberField
                label={t.detectionPanel.schedule.warmupMomentum}
                value={formData.training.warmup_momentum}
                onChange={(v) => setTraining({ warmup_momentum: v })}
                min={0}
                max={1}
                step={0.05}
              />
              <NumberField
                label={t.detectionPanel.schedule.warmupBiasLr}
                value={formData.training.warmup_bias_lr}
                onChange={(v) => setTraining({ warmup_bias_lr: v })}
                min={0}
                step={0.01}
              />
              <NumberField
                label={t.detectionPanel.schedule.boxGain}
                value={formData.training.box}
                onChange={(v) => setTraining({ box: v })}
                min={0}
                step={0.1}
                hint={t.detectionPanel.schedule.boxHint}
                help={paramHelp(t, "box")}
              />
              <NumberField
                label={t.detectionPanel.schedule.clsGain}
                value={formData.training.cls}
                onChange={(v) => setTraining({ cls: v })}
                min={0}
                step={0.1}
                hint={t.detectionPanel.schedule.clsHint}
                help={paramHelp(t, "cls")}
              />
              <NumberField
                label={t.detectionPanel.schedule.dflGain}
                value={formData.training.dfl}
                onChange={(v) => setTraining({ dfl: v })}
                min={0}
                step={0.1}
                hint={t.detectionPanel.schedule.dflHint}
                help={paramHelp(t, "dfl")}
              />
            </div>
          </div>

          {/* Regularization & mechanics */}
          <div style={card}>
            <div style={sectionLabel}>{t.detectionPanel.mechanics.title}</div>
            <div style={grid}>
              <NumberField
                label={t.detectionPanel.mechanics.labelSmoothing}
                value={formData.training.label_smoothing}
                onChange={(v) => setTraining({ label_smoothing: v })}
                min={0}
                max={1}
                step={0.01}
                help={paramHelp(t, "label_smoothing")}
                emptyValue={0}
              />
              <NumberField
                label={t.detectionPanel.mechanics.dropout}
                value={formData.training.dropout}
                onChange={(v) => setTraining({ dropout: v })}
                min={0}
                max={1}
                step={0.05}
                help={paramHelp(t, "dropout")}
                emptyValue={0}
              />
              <NumberField
                label={t.detectionPanel.mechanics.nbs}
                value={formData.training.nbs}
                onChange={(v) => setTraining({ nbs: Math.round(v) })}
                min={1}
                step={1}
                help={paramHelp(t, "nbs")}
              />
              <NumberField
                label={t.detectionPanel.mechanics.freeze}
                value={formData.training.freeze}
                onChange={(v) => setTraining({ freeze: Math.round(v) })}
                min={0}
                step={1}
                hint={t.detectionPanel.mechanics.freezeHint}
                help={paramHelp(t, "freeze")}
              />
              <NumberField
                label={t.detectionPanel.mechanics.closeMosaic}
                value={formData.training.close_mosaic}
                onChange={(v) => setTraining({ close_mosaic: Math.round(v) })}
                min={0}
                step={1}
                hint={t.detectionPanel.mechanics.closeMosaicHint}
                help={paramHelp(t, "close_mosaic")}
              />
              <Toggle
                label={t.detectionPanel.mechanics.amp}
                value={formData.training.amp}
                onChange={(v) => setTraining({ amp: v })}
                hint={t.detectionPanel.mechanics.ampHint}
                help={paramHelp(t, "amp")}
              />
              <Toggle
                label={t.detectionPanel.mechanics.singleCls}
                value={formData.training.single_cls}
                onChange={(v) => setTraining({ single_cls: v })}
                hint={t.detectionPanel.mechanics.singleClsHint}
                help={paramHelp(t, "single_cls")}
              />
              <Toggle
                label={t.detectionPanel.mechanics.rect}
                value={formData.training.rect}
                onChange={(v) => setTraining({ rect: v })}
                hint={t.detectionPanel.mechanics.rectHint}
                help={paramHelp(t, "rect")}
              />
              <Toggle
                label={t.detectionPanel.mechanics.multiScale}
                value={formData.training.multi_scale}
                onChange={(v) => setTraining({ multi_scale: v })}
                hint={t.detectionPanel.mechanics.multiScaleHint}
                help={paramHelp(t, "multi_scale")}
              />
            </div>
          </div>
        </>
      )}

      {/* Dataset — depois dos cards de treinamento (ordem canônica ADR-059) */}
      <div style={card}>
        <div style={sectionLabel}>{t.detectionPanel.dataset.title}</div>
        <div style={{ marginBottom: 14, maxWidth: 360 }}>
          <Segmented
            label={t.detectionPanel.dataset.source}
            value={formData.data.source}
            onChange={(v) =>
              setData({ source: v as DetectionForm["data"]["source"] })
            }
            options={[
              { value: "folder", label: t.detectionPanel.dataset.folderOption },
              { value: "yaml", label: t.detectionPanel.dataset.yamlOption },
            ]}
            hint={
              formData.data.source === "folder"
                ? t.detectionPanel.dataset.folderHint
                : t.detectionPanel.dataset.yamlHint
            }
          />
        </div>
        <div style={grid}>
          {formData.data.source === "folder" ? (
            <div style={{ gridColumn: "1 / -1", display: "flex", gap: 10, alignItems: "flex-end" }}>
              <div style={{ flex: 1 }}>
                <TextField
                  label={t.detectionPanel.dataset.baseDir}
                  value={formData.data.base_dir}
                  onChange={(v) => setData({ base_dir: v })}
                  placeholder={t.detectionPanel.dataset.baseDirPlaceholder}
                  hint={t.detectionPanel.dataset.baseDirHint}
                  mono
                />
              </div>
              <button
                type="button"
                onClick={() => void onPickFolder()}
                disabled={picking}
                style={pickButton}
              >
                {picking ? "…" : t.detectionPanel.dataset.browseFolder}
              </button>
            </div>
          ) : (
            <div style={{ gridColumn: "1 / -1", display: "flex", gap: 10, alignItems: "flex-end" }}>
              <div style={{ flex: 1 }}>
                <TextField
                  label={t.detectionPanel.dataset.yamlFile}
                  value={formData.data.data_yaml}
                  onChange={(v) => setData({ data_yaml: v })}
                  placeholder={t.detectionPanel.dataset.yamlPlaceholder}
                  hint={t.detectionPanel.dataset.yamlFileHint}
                  mono
                />
              </div>
              <button
                type="button"
                onClick={() => void onPickYaml()}
                disabled={picking}
                style={pickButton}
              >
                {picking ? "…" : t.detectionPanel.dataset.browseYaml}
              </button>
            </div>
          )}
          <NumberField
            label={t.detectionPanel.dataset.imageSize}
            value={formData.data.image_size}
            onChange={(v) => setData({ image_size: Math.round(v) })}
            min={32}
            step={32}
            suffix="px"
            hint={t.detectionPanel.dataset.imageSizeHint}
            help={paramHelp(t, "image_size")}
          />
        </div>
        <div style={{ ...grid, marginTop: 14 }}>
          <NumberField
            label={t.detectionPanel.dataset.numClasses}
            value={formData.model.num_classes}
            onChange={(v) => setModel({ num_classes: Math.round(v) })}
            min={1}
            step={1}
            hint={t.detectionPanel.dataset.numClassesHint}
          />
        </div>
        {formData.data.source === "folder" ? (
          <DetectionDatasetStats
            baseDir={formData.data.base_dir}
            onApplyClasses={(n) => setModel({ num_classes: n })}
          />
        ) : (
          <p
            style={{
              marginTop: 14,
              fontFamily: "var(--font-mono)",
              fontSize: 11,
              lineHeight: 1.6,
              color: "var(--vf-text-muted)",
            }}
          >
            <Rich text={t.detectionPanel.dataset.yamlNote} />
          </p>
        )}
      </div>

      {/* Pré-processamento (ADR-084). Ultralytics é dona do próprio loader, então
          os filtros são aplicados uma vez numa cópia temporária que o data.yaml
          passa a apontar — e a cópia é apagada quando o treino termina. */}
      <div style={card}>
        <PreprocessingPanel
          baseDir={formData.data.base_dir}
          steps={formData.data.preprocessing}
          onChange={(steps) => setData({ preprocessing: steps })}
        />
        {formData.data.preprocessing.length > 0 && (
          <p
            style={{
              marginTop: 12,
              fontFamily: "var(--font-mono)",
              fontSize: 11,
              lineHeight: 1.6,
              color: "var(--vf-text-muted)",
            }}
          >
            {t.detectionPanel.dataset.preprocessingNote}
          </p>
        )}
      </div>

      {isUltralytics && (
        <>
          {/* Augmentation */}
          <div style={card}>
            <div style={sectionLabel}>{t.detectionPanel.augmentation.title}</div>
            <Toggle
              label={t.detectionPanel.augmentation.toggle}
              value={formData.training.augmentation.augment}
              onChange={(v) => setAug({ augment: v })}
              hint={t.detectionPanel.augmentation.toggleHint}
            />
            {!formData.training.augmentation.augment && (
              <div
                style={{
                  marginTop: 14,
                  fontFamily: "var(--font-mono)",
                  fontSize: 11,
                  color: "var(--vf-text-muted)",
                }}
              >
                {t.paramPanel.hiddenParams(15)}
              </div>
            )}
            <div
              style={{
                ...grid,
                marginTop: 14,
                display: formData.training.augmentation.augment ? grid.display : "none",
              }}
            >
              <NumberField
                label={t.detectionPanel.augmentation.hsvHue}
                value={formData.training.augmentation.hsv_h}
                onChange={(v) => setAug({ hsv_h: v })}
                min={0}
                max={1}
                step={0.005}
              />
              <NumberField
                label={t.detectionPanel.augmentation.hsvSaturation}
                value={formData.training.augmentation.hsv_s}
                onChange={(v) => setAug({ hsv_s: v })}
                min={0}
                max={1}
                step={0.05}
              />
              <NumberField
                label={t.detectionPanel.augmentation.hsvValue}
                value={formData.training.augmentation.hsv_v}
                onChange={(v) => setAug({ hsv_v: v })}
                min={0}
                max={1}
                step={0.05}
              />
              <NumberField
                label={t.detectionPanel.augmentation.degrees}
                value={formData.training.augmentation.degrees}
                onChange={(v) => setAug({ degrees: v })}
                min={-180}
                max={180}
                step={1}
                suffix="°"
              />
              <NumberField
                label={t.detectionPanel.augmentation.translate}
                value={formData.training.augmentation.translate}
                onChange={(v) => setAug({ translate: v })}
                min={0}
                max={1}
                step={0.05}
              />
              <NumberField
                label={t.detectionPanel.augmentation.scale}
                value={formData.training.augmentation.scale}
                onChange={(v) => setAug({ scale: v })}
                min={0}
                step={0.05}
              />
              <NumberField
                label={t.detectionPanel.augmentation.shear}
                value={formData.training.augmentation.shear}
                onChange={(v) => setAug({ shear: v })}
                min={-180}
                max={180}
                step={1}
                suffix="°"
              />
              <NumberField
                label={t.detectionPanel.augmentation.perspective}
                value={formData.training.augmentation.perspective}
                onChange={(v) => setAug({ perspective: v })}
                min={0}
                max={0.001}
                step={0.0001}
              />
              <NumberField
                label={t.detectionPanel.augmentation.flipUpDown}
                value={formData.training.augmentation.flipud}
                onChange={(v) => setAug({ flipud: v })}
                min={0}
                max={1}
                step={0.05}
                hint={t.detectionPanel.augmentation.probability}
              />
              <NumberField
                label={t.detectionPanel.augmentation.flipLeftRight}
                value={formData.training.augmentation.fliplr}
                onChange={(v) => setAug({ fliplr: v })}
                min={0}
                max={1}
                step={0.05}
                hint={t.detectionPanel.augmentation.probability}
              />
              <NumberField
                label={t.detectionPanel.augmentation.bgrSwap}
                value={formData.training.augmentation.bgr}
                onChange={(v) => setAug({ bgr: v })}
                min={0}
                max={1}
                step={0.05}
                hint={t.detectionPanel.augmentation.probability}
              />
              <NumberField
                label={t.detectionPanel.augmentation.mosaic}
                value={formData.training.augmentation.mosaic}
                onChange={(v) => setAug({ mosaic: v })}
                min={0}
                max={1}
                step={0.05}
                hint={t.detectionPanel.augmentation.probability}
              />
              <NumberField
                label={t.detectionPanel.augmentation.mixup}
                value={formData.training.augmentation.mixup}
                onChange={(v) => setAug({ mixup: v })}
                min={0}
                max={1}
                step={0.05}
                hint={t.detectionPanel.augmentation.probability}
              />
              <NumberField
                label={t.detectionPanel.augmentation.copyPaste}
                value={formData.training.augmentation.copy_paste}
                onChange={(v) => setAug({ copy_paste: v })}
                min={0}
                max={1}
                step={0.05}
                hint={t.detectionPanel.augmentation.probability}
              />
              <SelectField
                label={t.detectionPanel.augmentation.autoAugment}
                value={formData.training.augmentation.auto_augment}
                onChange={(v) =>
                  setAug({
                    auto_augment:
                      v as DetectionAugmentationForm["auto_augment"],
                  })
                }
                options={[
                  { value: "randaugment", label: "RandAugment" },
                  { value: "autoaugment", label: "AutoAugment" },
                  { value: "augmix", label: "AugMix" },
                  { value: "none", label: t.detectionPanel.augmentation.disabled },
                ]}
              />
              <NumberField
                label={t.detectionPanel.augmentation.randomErasing}
                value={formData.training.augmentation.erasing}
                onChange={(v) => setAug({ erasing: v })}
                min={0}
                max={1}
                step={0.05}
                hint={t.detectionPanel.augmentation.probability}
              />
            </div>
          </div>
        </>
      )}

    </div>
  );
}
