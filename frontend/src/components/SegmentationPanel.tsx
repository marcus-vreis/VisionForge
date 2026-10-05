import { paramHelp } from "../lib/param-help";
import { useT } from "../i18n/useT";
import { ModelAdvice } from "./ModelAdvice";
import { AdvancedFields } from "./AdvancedFields";
import { useState } from "react";
import { fetchTaskSchema, pickDatasetFolder } from "../api/client";
import {
  SEGMENTATION_LOSSES,
  SEGMENTATION_MODELS,
  buildSegmentationPayload,
  ignoreIndexCollides,
  segmentationFormFromPayload,
  type SegmentationForm,
} from "../lib/segmentation-models";
import { exportConfigToYaml, validateParsedConfig } from "../lib/yaml-config";
import type { ValidationError } from "../hooks/useExperiment";
import { NumberField, SelectField, Segmented, TextField, Toggle } from "./controls";
import { CvCard, type CvPayload } from "./CvCard";
import { ExperimentHeader, type PanelStrategy } from "./ExperimentHeader";
import { ReplicatesCard } from "./ReplicatesCard";
import { SegmentationDatasetStats } from "./TaskDatasetStats";
import { SweepCard, type SweepPayload } from "./SweepCard";
import { TransformsSection } from "./TransformsSection";
import type { ReplicatesPayload } from "../lib/replicates-form";

const SWEEP_PATH_HINTS = [
  "training.learning_rate",
  "training.batch_size",
  "model.name",
];

interface SegmentationPanelProps {
  formData: SegmentationForm;
  setFormData: (updater: (prev: SegmentationForm) => SegmentationForm) => void;
  accent: string;
  validationErrors: ValidationError[];
  busy?: boolean;
  onSweep?: (payload: SweepPayload) => void;
  onReplicates?: (payload: ReplicatesPayload) => void;
  /** Lets App label and route the main Treinar button by the active strategy. */
  onStrategyChange?: (strategy: PanelStrategy) => void;
  /** Incremented by Treinar so the selected strategy's card runs. */
  runSignal?: number;
  onCv?: (payload: CvPayload) => void;
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

/** Schema-aligned form for a paired image/mask semantic-segmentation run. */
export function SegmentationPanel({
  formData,
  setFormData,
  accent,
  validationErrors,
  busy,
  onSweep,
  onReplicates,
  onStrategyChange,
  runSignal,
  onCv,
}: SegmentationPanelProps) {
  const t = useT();
  const compareMetrics = [
    { value: "miou", label: "mIoU" },
    { value: "dice", label: "Dice" },
    { value: "pixel_acc", label: t.segmentationPanel.pixelAcc },
  ];
  const [picking, setPicking] = useState(false);
  const [strategy, setStrategy] = useState<PanelStrategy>("simple");

  const setModel = (patch: Partial<SegmentationForm["model"]>) =>
    setFormData((p) => ({ ...p, model: { ...p.model, ...patch } }));
  const setData = (patch: Partial<SegmentationForm["data"]>) =>
    setFormData((p) => ({ ...p, data: { ...p.data, ...patch } }));
  const setTraining = (patch: Partial<SegmentationForm["training"]>) =>
    setFormData((p) => ({ ...p, training: { ...p.training, ...patch } }));
  const setTransforms = (patch: Partial<SegmentationForm["transforms"]>) =>
    setFormData((p) => ({ ...p, transforms: { ...p.transforms, ...patch } }));
  const setPreprocessing = (steps: SegmentationForm["preprocessing"]) =>
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

  const collides = ignoreIndexCollides(formData);

  const card: React.CSSProperties = {
    background: "var(--vf-panel)",
    border: "1px solid var(--vf-panel-stroke)",
    borderRadius: 18,
    padding: 26,
    backdropFilter: "blur(14px)",
  };

  const warnBanner: React.CSSProperties = {
    padding: "12px 16px",
    background: "oklch(0.704 0.191 22.216 / 0.10)",
    border: "1px solid oklch(0.704 0.191 22.216 / 0.4)",
    borderRadius: 12,
    fontFamily: "var(--font-mono)",
    fontSize: 12,
    color: "oklch(0.85 0.14 22)",
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
      {validationErrors.length > 0 && (
        <div style={warnBanner}>
          {validationErrors.slice(0, 5).map((e, i) => (
            <div key={i}>
              {e.field.join(" › ")}: {e.message}
            </div>
          ))}
        </div>
      )}

      {collides && (
        <div style={warnBanner}>
          {t.segmentationPanel.ignoreIndexCollision(
            formData.data.ignore_index,
            formData.model.num_classes - 1,
          )}
        </div>
      )}

      {/* Cabeçalho canônico: nome + YAML + estratégia numa caixa (ADR-059) */}
      <ExperimentHeader
        name={formData.name}
        onNameChange={(v) => setFormData((p) => ({ ...p, name: v }))}
        placeholder={t.segmentationPanel.namePlaceholder}
        strategy={strategy}
        onStrategyChange={(s) => {
          setStrategy(s);
          onStrategyChange?.(s);
        }}
        strategies={[
          { value: "simple", label: t.paramPanel.blocks.simple },
          { value: "cv", label: t.paramPanel.blocks.crossValidation },
          { value: "sweep", label: t.taskPanel.sweep },
          { value: "replicates", label: t.taskPanel.replicates },
        ]}
        onExportYaml={() =>
          exportConfigToYaml(buildSegmentationPayload(formData), formData.name)
        }
        onImportConfig={async (data) => {
          try {
            const schema = await fetchTaskSchema("segmentation");
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
          setFormData(() => segmentationFormFromPayload(data));
          return null;
        }}
      />
      {strategy === "sweep" && onSweep && (
        <SweepCard
          metrics={compareMetrics}
          pathHints={SWEEP_PATH_HINTS}
          modelOptions={SEGMENTATION_MODELS}
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
      {strategy === "cv" && onCv && (
        <CvCard accent={accent} disabled={busy} onCv={onCv} runSignal={runSignal} />
      )}

      {/* Modelo */}
      <div style={card}>
        <div style={sectionLabel}>{t.segmentationPanel.model.title}</div>
        <div style={grid}>
          <SelectField
            label={t.segmentationPanel.model.architecture}
            value={formData.model.name}
            onChange={(v) => setModel({ name: v })}
            options={SEGMENTATION_MODELS}
            hint={t.segmentationPanel.model.architectureHint}
          />
          <NumberField
            label={t.segmentationPanel.model.numClasses}
            value={formData.model.num_classes}
            onChange={(v) => setModel({ num_classes: Math.round(v) })}
            min={1}
            step={1}
            hint={t.segmentationPanel.model.numClassesHint}
          />
          <Toggle
            label={t.taskPanel.pretrained}
            value={formData.model.pretrained}
            onChange={(v) => setModel({ pretrained: v })}
            hint={t.segmentationPanel.model.pretrainedHint}
          />
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
          <Segmented
            label={t.taskPanel.training.loss}
            value={formData.training.loss}
            onChange={(v) => setTraining({ loss: v })}
            options={SEGMENTATION_LOSSES}
            hint={t.segmentationPanel.lossHint}
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
        <AdvancedFields count={3}>
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
        <ModelAdvice
          architecture={formData.model.name}
          optimizer={formData.training.optimizer}
          learningRate={formData.training.learning_rate}
          baseDir={String(formData.data.base_dir || "") || undefined}
          pretrained={formData.model.pretrained !== false}
          onApply={(next) => {
            setTraining({
              optimizer: next.optimizer as typeof formData.training.optimizer,
              learning_rate: next.learning_rate,
            });
          }}
        />
      </div>

      {/* Transfer learning */}
      <div style={card}>
        <div style={sectionLabel}>{t.taskPanel.transfer.title}</div>
        <div style={grid}>
          <Segmented
            label={t.taskPanel.transfer.mode}
            value={formData.transfer}
            onChange={(v) =>
              setFormData((p) => ({
                ...p,
                transfer: v as SegmentationForm["transfer"],
              }))
            }
            options={[
              { value: "none", label: t.taskPanel.transfer.full },
              { value: "feature_extraction", label: t.taskPanel.transfer.featureExtraction },
              { value: "fine_tuning", label: t.taskPanel.transfer.fineTuning },
            ]}
            hint={t.segmentationPanel.transferHint}
          />
          {formData.transfer === "fine_tuning" && (
            <NumberField
              label={t.taskPanel.transfer.backboneLr}
              value={formData.backbone_lr_multiplier}
              onChange={(v) =>
                setFormData((p) => ({ ...p, backbone_lr_multiplier: v }))
              }
              min={0.0001}
              max={1}
              step={0.05}
              hint={t.taskPanel.transfer.backboneLrHint}
            />
          )}
        </div>
      </div>

      {/* Dataset */}
      <div style={card}>
        <div style={sectionLabel}>{t.segmentationPanel.dataset.title}</div>
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
                placeholder={t.segmentationPanel.dataset.baseDirPlaceholder}
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
            label={t.taskPanel.dataset.imagesSubdir}
            value={formData.data.images_subdir}
            onChange={(v) => setData({ images_subdir: v })}
            hint={t.segmentationPanel.dataset.imagesSubdirHint}
            mono
          />
          <TextField
            label={t.segmentationPanel.dataset.masksSubdir}
            value={formData.data.masks_subdir}
            onChange={(v) => setData({ masks_subdir: v })}
            hint={t.segmentationPanel.dataset.masksSubdirHint}
            mono
          />
          <TextField
            label={t.taskPanel.dataset.trainSplit}
            value={formData.data.train_dir}
            onChange={(v) => setData({ train_dir: v })}
            mono
          />
          <TextField
            label={t.taskPanel.dataset.valSplit}
            value={formData.data.val_dir}
            onChange={(v) => setData({ val_dir: v })}
            mono
          />
          <TextField
            label={t.taskPanel.dataset.testSplit}
            value={formData.data.test_dir}
            onChange={(v) => setData({ test_dir: v })}
            hint={t.taskPanel.dataset.optional}
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
          <NumberField
            label={t.segmentationPanel.dataset.ignoreIndex}
            value={formData.data.ignore_index}
            onChange={(v) => setData({ ignore_index: Math.round(v) })}
            step={1}
            hint={t.segmentationPanel.dataset.ignoreIndexHint}
          />
        </div>
        <SegmentationDatasetStats
          baseDir={formData.data.base_dir}
          imagesSubdir={formData.data.images_subdir}
          masksSubdir={formData.data.masks_subdir}
          trainDir={formData.data.train_dir}
          valDir={formData.data.val_dir}
          testDir={formData.data.test_dir}
          onApplyClasses={(n) => setModel({ num_classes: n })}
        />
      </div>

      {/* Pré-processamento + augmentação (ADR-059) */}
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
