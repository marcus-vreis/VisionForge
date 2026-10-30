import { paramHelp } from "../lib/param-help";
import { useT } from "../i18n/useT";
import { runPicker } from "../lib/picker-feedback";
import { PickerNote } from "./PickerNote";
import { sampleAnomalyForm } from "../lib/guides/sample-forms";
import { SampleDatasetButton } from "./SampleDatasetButton";
import { useState } from "react";
import { fetchTaskSchema, pickDatasetFolder } from "../api/client";
import {
  anomalyBackbones,
  anomalyModels,
  anomalyFormFromPayload,
  buildAnomalyPayload,
  isPatchCore,
  makeDefaultAnomalyForm,
  type AnomalyForm,
} from "../lib/anomaly-models";
import { checkImportedConfig, exportConfigToYaml } from "../lib/yaml-config";
import type { ValidationError } from "../hooks/useExperiment";
import { NumberField, SelectField, Segmented, TextField, Toggle } from "./controls";
import { AdvancedFields } from "./AdvancedFields";
import { StrategyCard, type PanelStrategy } from "./ExperimentHeader";
import { ReplicatesCard } from "./ReplicatesCard";
import { ReviewStep } from "./ReviewStep";
import { StepShell } from "./StepShell";
import { AnomalyDatasetStats } from "./TaskDatasetStats";
import { CompareModelsCard } from "./CompareModelsCard";
import { TransformsSection } from "./TransformsSection";
import { datasetLabel, suggestExperimentName } from "../lib/experiment-name";
import { useAutoExperimentName } from "../lib/use-auto-experiment-name";
import { countChanged, errorsUnder } from "../lib/steps";
import type { ReplicatesPayload } from "../lib/replicates-form";
import type { ComparePayload } from "../lib/compare-form";

interface AnomalyPanelProps {
  formData: AnomalyForm;
  setFormData: (updater: (prev: AnomalyForm) => AnomalyForm) => void;
  accent: string;
  validationErrors: ValidationError[];
  busy?: boolean;
  onReplicates?: (payload: ReplicatesPayload) => void;
  onCompare?: (payload: ComparePayload) => void;
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
  gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))",
  gap: 18,
};

const stack: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: 18,
};

/** Schema-aligned form for an MVTec-style anomaly-detection run. */
export function AnomalyPanel({
  formData,
  setFormData,
  accent,
  validationErrors,
  busy,
  onReplicates,
  onCompare,
  onStrategyChange,
  runSignal,
}: AnomalyPanelProps) {
  const t = useT();
  const compareMetrics = [
    { value: "auroc", label: "AUROC" },
    { value: "image_f1", label: t.anomalyPanel.imageF1 },
  ];
  const [picking, setPicking] = useState(false);
  const [pickNote, setPickNote] = useState<string | null>(null);
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
      setPickNote(
        await runPicker(
          pickDatasetFolder,
          (path) => setData({ base_dir: path }),
          { canceled: t.datasetPicker.cancelled, failed: t.datasetPicker.pickFailed },
        ),
      );
    } finally {
      setPicking(false);
    }
  };

  const patchcore = isPatchCore(formData);

  const defaults = makeDefaultAnomalyForm();
  // No sweep: anomaly has no labelled validation split to select on, and
  // the server refuses one rather than tune on the test set (ADR-131).
  // The comparison stays: it reports test metrics and tunes nothing.
  const strategies: { value: PanelStrategy; label: string }[] = [
    { value: "simple", label: t.paramPanel.blocks.simple },
    { value: "compare", label: t.anomalyPanel.compareLabel },
    { value: "replicates", label: t.taskPanel.replicates },
  ];
  const strategyLabel = strategies.find((s) => s.value === strategy)?.label ?? "";

  const suggestion = suggestExperimentName({
    datasetPath: formData.data.base_dir,
    model: formData.model.name,
    epochs: formData.training.epochs,
    learningRate: formData.training.learning_rate,
    strategy,
  });
  useAutoExperimentName(
    "anomaly",
    formData.name,
    suggestion,
    (name) => setFormData((p) => ({ ...p, name })),
    [defaults.name],
  );

  const card: React.CSSProperties = {
    background: "var(--vf-panel)",
    border: "1px solid var(--vf-panel-stroke)",
    borderRadius: 18,
    padding: 26,
    backdropFilter: "blur(14px)",
  };

  const f = formData;
  const d = defaults;
  const row = (label: string, value: string, differs?: boolean) => ({
    label,
    value,
    changed: differs,
  });

  const dataStep = (
    <div style={stack}>
      <div style={card}>
        <div style={sectionLabel}>{t.anomalyPanel.dataset.title}</div>
        <div style={grid}>
          <div
            data-tour="dataset"
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
          <PickerNote text={pickNote} />
          <SampleDatasetButton
            task="anomaly"
            onApply={(path, setup) => setFormData((p) => sampleAnomalyForm(p, path, setup))}
          />
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
        </div>
        <div data-tour="dataset-stats">
          <AnomalyDatasetStats
            baseDir={formData.data.base_dir}
            trainDir={formData.data.train_dir}
            testDir={formData.data.test_dir}
            normalDir={formData.data.normal_dir}
          />
        </div>
      </div>
    </div>
  );

  const strategyStep = (
    <div style={stack}>
      <StrategyCard
        strategy={strategy}
        onStrategyChange={(s) => {
          setStrategy(s);
          onStrategyChange?.(s);
        }}
        strategies={strategies}
      />
      {strategy === "compare" && onCompare && (
        <CompareModelsCard
          models={anomalyModels(t)}
          metrics={compareMetrics}
          note={t.anomalyPanel.compareNote}
          accent={accent}
          disabled={busy}
          onCompare={onCompare}
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
    </div>
  );

  const modelStep = (
    <div style={stack}>
      <div style={card}>
        <div style={sectionLabel}>{t.anomalyPanel.model.title}</div>
        <div style={grid}>
          <Segmented
            label={t.anomalyPanel.model.method}
            value={formData.model.name}
            onChange={(v) => setModel({ name: v })}
            options={anomalyModels(t)}
            hint={t.anomalyPanel.model.methodHint}
          />
          {patchcore ? (
            <>
              <SelectField
                label={t.taskPanel.backbone}
                value={formData.model.backbone}
                onChange={(v) => setModel({ backbone: v })}
                options={anomalyBackbones(t)}
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
    </div>
  );

  const trainingStep = (
    <div style={stack}>
      <div style={card}>
        <div style={sectionLabel}>{t.taskPanel.training.title}</div>
        <div data-tour="training" style={grid}>
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
        <AdvancedFields count={4} flush>
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
    </div>
  );

  const imageStep = (
    <div style={stack}>
      <div style={card}>
        <div style={grid}>
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

  const reviewStep = (
    <ReviewStep
      taskKey="anomaly"
      name={f.name}
      onNameChange={(name) => setFormData((p) => ({ ...p, name }))}
      suggestion={suggestion}
      onExportYaml={() => exportConfigToYaml(buildAnomalyPayload(formData), formData.name)}
      onImportConfig={async (data) => {
        let payload = data;
        try {
          const schema = await fetchTaskSchema("anomaly");
          const checked = checkImportedConfig(t, data, schema, schema.$defs ?? {});
          if ("problem" in checked) return checked.problem;
          payload = checked.data;
        } catch {
          // schema unavailable → import tolerantly; o 422 do submit cobre.
        }
        setFormData(() => anomalyFormFromPayload(payload));
        return null;
      }}
      sections={[
        {
          step: "data",
          rows: [
            row(t.taskPanel.dataset.baseDir, f.data.base_dir || "—"),
            row(t.taskPanel.dataset.trainSplit, f.data.train_dir, f.data.train_dir !== d.data.train_dir),
            row(t.taskPanel.dataset.testSplit, f.data.test_dir, f.data.test_dir !== d.data.test_dir),
            row(t.anomalyPanel.dataset.normalDir, f.data.normal_dir, f.data.normal_dir !== d.data.normal_dir),
          ],
        },
        {
          step: "strategy",
          rows: [row(t.experimentHeader.strategyTitle, strategyLabel, strategy !== "simple")],
        },
        {
          step: "model",
          rows: [
            row(t.anomalyPanel.model.method, f.model.name, f.model.name !== d.model.name),
            ...(patchcore
              ? [
                  row(t.taskPanel.backbone, f.model.backbone, f.model.backbone !== d.model.backbone),
                  row(
                    t.anomalyPanel.model.coresetRatio,
                    String(f.model.coreset_ratio),
                    f.model.coreset_ratio !== d.model.coreset_ratio,
                  ),
                ]
              : [
                  row(
                    t.anomalyPanel.model.latentDim,
                    String(f.model.latent_dim),
                    f.model.latent_dim !== d.model.latent_dim,
                  ),
                ]),
          ],
        },
        {
          step: "training",
          rows: [
            row(t.taskPanel.training.epochs, String(f.training.epochs), f.training.epochs !== d.training.epochs),
            row(
              t.taskPanel.training.learningRate,
              String(f.training.learning_rate),
              f.training.learning_rate !== d.training.learning_rate,
            ),
            row(
              t.taskPanel.training.batchSize,
              String(f.training.batch_size),
              f.training.batch_size !== d.training.batch_size,
            ),
            row(t.taskPanel.training.seed, String(f.training.seed), f.training.seed !== d.training.seed),
            row(
              t.anomalyPanel.threshold,
              String(f.training.threshold_percentile),
              f.training.threshold_percentile !== d.training.threshold_percentile,
            ),
          ],
        },
        {
          step: "image",
          rows: [
            row(
              t.taskPanel.dataset.imageSize,
              `${f.data.image_size} px`,
              f.data.image_size !== d.data.image_size,
            ),
          ],
        },
      ]}
      errors={validationErrors.map((e) => `${e.field.join(" › ")}: ${e.message}`)}
      ready={f.data.base_dir.trim() !== ""}
    />
  );

  return (
    <StepShell
      taskKey="anomaly"
      steps={[
        {
          id: "data",
          summary: datasetLabel(f.data.base_dir) || undefined,
          missing: f.data.base_dir.trim() === "",
          errors: errorsUnder(validationErrors, [
            "data.base_dir",
            "data.train_dir",
            "data.test_dir",
            "data.normal_dir",
          ]),
          content: dataStep,
        },
        { id: "strategy", summary: strategyLabel, content: strategyStep },
        {
          id: "model",
          summary: f.model.name,
          changes: countChanged(f, d, [
            "model.name",
            "model.backbone",
            "model.latent_dim",
            "model.coreset_ratio",
            "model.pretrained",
          ]),
          errors: errorsUnder(validationErrors, ["model"]),
          content: modelStep,
        },
        {
          id: "training",
          summary: `e${f.training.epochs} · lr ${f.training.learning_rate}`,
          changes: countChanged(f, d, [
            "training.epochs",
            "training.batch_size",
            "training.learning_rate",
            "training.seed",
            "training.threshold_percentile",
            "training.optimizer",
            "training.early_stopping_patience",
            "training.deterministic",
          ]),
          errors: errorsUnder(validationErrors, ["training"]),
          content: trainingStep,
        },
        {
          id: "image",
          summary: `${f.data.image_size} px`,
          changes: countChanged(f, d, ["data.image_size", "transforms", "preprocessing"]),
          errors: errorsUnder(validationErrors, [
            "data.image_size",
            "data.transforms",
            "data.preprocessing",
          ]),
          content: imageStep,
        },
        { id: "review", content: reviewStep },
      ]}
    />
  );
}
