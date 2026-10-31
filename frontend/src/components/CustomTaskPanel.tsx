import { useEffect, useState } from "react";

import { fetchCustomSchema } from "../api/client";
import { useT } from "../i18n/useT";
import {
  buildCustomForm,
  metricOptions,
  type CustomTaskDefinition,
} from "../lib/custom-tasks";
import { exportConfigToYaml } from "../lib/yaml-config";
import { suggestExperimentName } from "../lib/experiment-name";
import { resolveSchema, visibleChildren } from "../lib/schema-form";
import { countChanged, errorsUnder, type StepId } from "../lib/steps";
import { useAutoExperimentName } from "../lib/use-auto-experiment-name";
import type { ValidationError } from "../hooks/useExperiment";
import type { JsonSchema } from "../types/schema";
import { StrategyCard, type PanelStrategy } from "./ExperimentHeader";
import { CustomTaskManageCard } from "./CustomTaskManageCard";
import { Rich } from "./Rich";
import { ReplicatesCard } from "./ReplicatesCard";
import { ReviewStep } from "./ReviewStep";
import { SchemaForm } from "./SchemaForm";
import { StepShell, type StepDef } from "./StepShell";
import { SweepCard, type SweepPayload } from "./SweepCard";
import type { ReplicatesPayload } from "../lib/replicates-form";

/**
 * The panel a researcher-defined task gets for free (ADR-058).
 *
 * Same step layout as every built-in panel (ADR-156), except the form is
 * generated from `GET /api/custom/{key}/schema` instead of being hand-written:
 * each top-level block of the schema lands on the step it belongs to. Custom
 * tasks have no K-fold endpoint, so the strategy selector offers exactly what
 * the API supports.
 */

interface CustomTaskPanelProps {
  task: CustomTaskDefinition;
  formData: Record<string, unknown>;
  setFormData: (next: Record<string, unknown>) => void;
  validationErrors: ValidationError[];
  busy?: boolean;
  onSweep: (payload: SweepPayload) => void;
  onReplicates: (payload: ReplicatesPayload) => void;
  /** Lets App label and route the main Treinar button by the active strategy. */
  onStrategyChange?: (strategy: PanelStrategy) => void;
  /** Incremented by Treinar so the selected strategy's card runs. */
  runSignal?: number;
  /** Refetch the tab list after this task was hidden or deleted. */
  onRemoved?: () => void;
}

const noticeStyle: React.CSSProperties = {
  padding: "14px 18px",
  background: "var(--vf-panel)",
  border: "1px solid var(--vf-panel-stroke)",
  borderRadius: 14,
  fontFamily: "var(--font-mono)",
  fontSize: 12,
  color: "var(--vf-text-dim)",
  lineHeight: 1.6,
};

/** The step a top-level schema field belongs to. The blocks every custom task
 *  inherits map to their step; the task's own scalars and anything unknown go
 *  to training, where the researcher looks for "the rest". */
function stepOfField(name: string, isObject: boolean): StepId {
  if (!isObject) return "training";
  if (name === "data") return "data";
  if (name === "model") return "model";
  if (name === "transforms" || name === "preprocessing" || name.startsWith("augment")) {
    return "image";
  }
  return "training";
}

const OMITTED = ["name", "device"];

const num = (v: unknown): number | undefined =>
  typeof v === "number" && Number.isFinite(v) ? v : undefined;
const str = (v: unknown): string | undefined =>
  typeof v === "string" && v.trim() !== "" ? v : undefined;
const at = (form: Record<string, unknown>, block: string, key: string): unknown => {
  const b = form[block];
  return b != null && typeof b === "object" ? (b as Record<string, unknown>)[key] : undefined;
};

const stack: React.CSSProperties = { display: "flex", flexDirection: "column", gap: 18 };

export function CustomTaskPanel({
  task,
  formData,
  setFormData,
  validationErrors,
  busy,
  onSweep,
  onReplicates,
  onStrategyChange,
  runSignal,
  onRemoved,
}: CustomTaskPanelProps) {
  const t = useT();
  const [schema, setSchema] = useState<JsonSchema | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Re-fetch per task: switching tabs must not show the previous task's form.
  useEffect(() => {
    let cancelled = false;
    setSchema(null);
    setError(null);
    fetchCustomSchema(task.key)
      .then((s) => {
        if (cancelled) return;
        setSchema(s);
        // Only seed defaults when the form is still empty, so a tab switch
        // back does not discard what the researcher typed.
        setFormData(
          Object.keys(formData).length > 0 ? formData : buildCustomForm(s),
        );
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
    };
    // formData is intentionally out of the dep list: it changes on every
    // keystroke and would refetch the schema each time.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [task.key, setFormData]);

  if (error) {
    return (
      <div style={{ ...noticeStyle, color: "oklch(calc(0.85 - var(--lt-flip, 0) * 0.4) 0.14 22)" }}>
        {t.customTaskPanel.schemaLoadFailed} <strong>{task.key}</strong>: {error}
        <br />
        <Rich text={t.customTaskPanel.schemaLoadHint} />
      </div>
    );
  }

  if (!schema) {
    return <div style={noticeStyle}>{t.customTaskPanel.loadingForm(task.label)}</div>;
  }

  // A child so the auto-name hook only runs once the form is seeded: naming an
  // empty form would make it look filled and skip the schema defaults.
  return (
    <CustomTaskSteps
      task={task}
      schema={schema}
      formData={formData}
      setFormData={setFormData}
      validationErrors={validationErrors}
      busy={busy}
      onSweep={onSweep}
      onReplicates={onReplicates}
      onStrategyChange={onStrategyChange}
      runSignal={runSignal}
      onRemoved={onRemoved}
    />
  );
}

export function CustomTaskSteps({
  task,
  schema,
  formData,
  setFormData,
  validationErrors,
  busy,
  onSweep,
  onReplicates,
  onStrategyChange,
  runSignal,
  onRemoved,
}: CustomTaskPanelProps & { schema: JsonSchema }) {
  const t = useT();
  const [strategy, setStrategy] = useState<PanelStrategy>("simple");
  // App keys the step of every task by its tab key (`activeKey`), which for
  // a researcher's task is its bare key: the Treinar lock reads that entry.
  const taskKey = task.key;

  // Custom tasks have no K-fold endpoint, so the selector offers exactly what the API supports.
  const strategies: { value: PanelStrategy; label: string }[] = [
    { value: "simple", label: t.paramPanel.blocks.simple },
    { value: "sweep", label: t.taskPanel.sweep },
    { value: "replicates", label: t.taskPanel.replicates },
  ];
  const strategyLabel = strategies.find((s) => s.value === strategy)?.label ?? "";
  const metrics = metricOptions(task);
  const name = typeof formData.name === "string" ? formData.name : "";
  const defaults = buildCustomForm(schema);

  const defs = schema.$defs ?? {};
  // Which top-level fields each step gets; a step with none is not shown.
  const fieldsByStep = new Map<StepId, string[]>();
  for (const [field, child] of visibleChildren(schema, defs)) {
    if (OMITTED.includes(field)) continue;
    const step = stepOfField(field, resolveSchema(child, defs).type === "object");
    fieldsByStep.set(step, [...(fieldsByStep.get(step) ?? []), field]);
  }
  const fieldsOf = (step: StepId) => fieldsByStep.get(step) ?? [];
  const form = (step: StepId) => (
    <SchemaForm
      schema={schema}
      value={formData}
      onChange={setFormData}
      validationErrors={validationErrors}
      // `name` lives in the Revisão step; `device` is owned by the
      // DeviceSelector in the bottom bar (same rule as the built-in panels).
      omit={OMITTED}
      include={(field, isObject) => stepOfField(field, isObject) === step}
    />
  );
  const stepStats = (step: StepId) => {
    const fields = fieldsOf(step);
    return {
      changes: countChanged(formData, defaults, fields),
      errors: errorsUnder(validationErrors, fields),
    };
  };

  const epochs = num(at(formData, "training", "epochs"));
  const lr = num(at(formData, "training", "learning_rate"));
  const modelName = str(at(formData, "model", "name"));
  const suggestion = suggestExperimentName({
    datasetPath:
      str(at(formData, "data", "base_dir")) ?? str(at(formData, "data", "root")) ?? "",
    model: modelName,
    epochs,
    learningRate: lr,
    strategy,
  });
  const defaultNames = [`${task.key}_001`];
  if (typeof defaults.name === "string") defaultNames.push(defaults.name);
  useAutoExperimentName(
    taskKey,
    name,
    suggestion,
    (next) => setFormData({ ...formData, name: next }),
    defaultNames,
  );

  const keyRows: Partial<Record<StepId, { label: string; block: string; key: string }[]>> = {
    data: [{ label: t.taskPanel.dataset.baseDir, block: "data", key: "base_dir" }],
    model: [{ label: t.taskPanel.backbone, block: "model", key: "name" }],
    training: [
      { label: t.taskPanel.training.epochs, block: "training", key: "epochs" },
      { label: t.taskPanel.training.learningRate, block: "training", key: "learning_rate" },
      { label: t.taskPanel.training.batchSize, block: "training", key: "batch_size" },
      { label: t.taskPanel.training.seed, block: "training", key: "seed" },
    ],
  };
  const reviewRows = (step: StepId) =>
    (keyRows[step] ?? [])
      .filter((r) => at(formData, r.block, r.key) !== undefined)
      .map((r) => ({
        label: r.label,
        value: String(at(formData, r.block, r.key)),
        changed:
          JSON.stringify(at(formData, r.block, r.key)) !==
          JSON.stringify(at(defaults, r.block, r.key)),
      }));

  const steps: StepDef[] = [];
  if (fieldsOf("data").length > 0) {
    steps.push({ id: "data", errors: stepStats("data").errors, content: form("data") });
  }
  steps.push({
    id: "strategy",
    summary: strategyLabel,
    content: (
      <div style={stack}>
        <StrategyCard
          strategy={strategy}
          onStrategyChange={(s) => {
            setStrategy(s);
            onStrategyChange?.(s);
          }}
          strategies={strategies}
        />
        {strategy === "sweep" && (
          <SweepCard
            metrics={metrics}
            pathHints={["training.learning_rate", "training.batch_size"]}
            accent={task.accent}
            disabled={busy}
            onSweep={onSweep}
            runSignal={runSignal}
          />
        )}
        {strategy === "replicates" && (
          <ReplicatesCard
            metrics={metrics}
            accent={task.accent}
            disabled={busy}
            onReplicates={onReplicates}
            runSignal={runSignal}
          />
        )}
      </div>
    ),
  });
  if (fieldsOf("model").length > 0) {
    steps.push({ id: "model", summary: modelName, ...stepStats("model"), content: form("model") });
  }
  if (fieldsOf("training").length > 0) {
    steps.push({
      id: "training",
      summary: epochs != null && lr != null ? `e${epochs} · lr ${lr}` : undefined,
      ...stepStats("training"),
      content: form("training"),
    });
  }
  if (fieldsOf("image").length > 0) {
    steps.push({ id: "image", ...stepStats("image"), content: form("image") });
  }
  steps.push({
    id: "review",
    content: (
      <div style={stack}>
        <ReviewStep
          taskKey={taskKey}
          name={name}
          onNameChange={(v) => setFormData({ ...formData, name: v })}
          suggestion={suggestion}
          onExportYaml={() => exportConfigToYaml(formData, name || task.key)}
          onImportConfig={(data) => {
            // The task's own Config validates on submit (422 with field paths),
            // so import stays tolerant here rather than duplicating the rules.
            setFormData(data);
            return null;
          }}
          sections={[
            { step: "data", rows: reviewRows("data") },
            {
              step: "strategy",
              rows: [
                {
                  label: t.experimentHeader.strategyTitle,
                  value: strategyLabel,
                  changed: strategy !== "simple",
                },
              ],
            },
            { step: "model", rows: reviewRows("model") },
            { step: "training", rows: reviewRows("training") },
          ]}
          errors={validationErrors.map((e) => `${e.field.join(" › ")}: ${e.message}`)}
          ready
        />
        {/* Last in the Revisão on purpose: you reach it by scrolling past the
            thing you came here to configure, not by aiming near it. */}
        {onRemoved && (
          <CustomTaskManageCard
            taskKey={task.key}
            label={task.label}
            onRemoved={onRemoved}
          />
        )}
      </div>
    ),
  });

  return <StepShell taskKey={taskKey} steps={steps} />;
}
