import { useEffect, useState } from "react";

import { fetchCustomSchema } from "../api/client";
import {
  buildCustomForm,
  metricOptions,
  type CustomTaskDefinition,
} from "../lib/custom-tasks";
import { exportConfigToYaml } from "../lib/yaml-config";
import type { ValidationError } from "../hooks/useExperiment";
import type { JsonSchema } from "../types/schema";
import { ExperimentHeader, type PanelStrategy } from "./ExperimentHeader";
import { CustomTaskManageCard } from "./CustomTaskManageCard";
import { ReplicatesCard } from "./ReplicatesCard";
import { SchemaForm } from "./SchemaForm";
import { SweepCard, type SweepPayload } from "./SweepCard";
import type { ReplicatesPayload } from "../lib/replicates-form";

/**
 * The panel a researcher-defined task gets for free (ADR-058, brick 6).
 *
 * Same canonical contract as every built-in panel (ADR-059): the experiment
 * header with name + YAML export/import + strategy selector, then the form —
 * except the form is generated from `GET /api/custom/{key}/schema` instead of
 * being hand-written. Custom tasks have no K-fold endpoint, so the strategy
 * selector offers exactly what the API supports.
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

const STRATEGIES: { value: PanelStrategy; label: string }[] = [
  { value: "simple", label: "Treino simples" },
  { value: "sweep", label: "Sweep" },
  { value: "replicates", label: "Réplicas" },
];

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
  const [schema, setSchema] = useState<JsonSchema | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [strategy, setStrategy] = useState<PanelStrategy>("simple");

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

  const metrics = metricOptions(task);
  const name = typeof formData.name === "string" ? formData.name : "";

  if (error) {
    return (
      <div style={{ ...noticeStyle, color: "oklch(0.85 0.14 22)" }}>
        Não foi possível carregar o schema de <strong>{task.key}</strong>: {error}
        <br />
        Verifique o arquivo em <code>user_tasks/</code> — um erro de import é
        registrado no log do servidor e a tarefa fica sem formulário.
      </div>
    );
  }

  if (!schema) {
    return <div style={noticeStyle}>Carregando o formulário de {task.label}…</div>;
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
      <ExperimentHeader
        name={name}
        onNameChange={(v) => setFormData({ ...formData, name: v })}
        placeholder={`${task.key}_001`}
        strategy={strategy}
        onStrategyChange={(s) => {
          setStrategy(s);
          onStrategyChange?.(s);
        }}
        strategies={STRATEGIES}
        onExportYaml={() => exportConfigToYaml(formData, name || task.key)}
        onImportConfig={(data) => {
          // The task's own Config validates on submit (422 with field paths),
          // so import stays tolerant here rather than duplicating the rules.
          setFormData(data);
          return null;
        }}
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

      <SchemaForm
        schema={schema}
        value={formData}
        onChange={setFormData}
        validationErrors={validationErrors}
        // `name` lives in the header above; `device` is owned by the
        // DeviceSelector in the bottom bar (same rule as the built-in panels).
        omit={["name", "device"]}
      />

      {/* Last in the panel on purpose: you reach it by scrolling past the
          thing you came here to configure, not by aiming near it. */}
      {onRemoved && (
        <CustomTaskManageCard
          taskKey={task.key}
          label={task.label}
          onRemoved={onRemoved}
        />
      )}
    </div>
  );
}
