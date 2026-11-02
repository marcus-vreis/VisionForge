import type { LearningRatePlanRequest } from "../api/client";
import { useT } from "../i18n/useT";
import { shortKey } from "../lib/experiment-name";
import { useStepNav, type StepId } from "../lib/steps";
import { ExperimentNameCard } from "./ExperimentHeader";
import { LearningRatePlanCard } from "./LearningRatePlanCard";

export interface ReviewRow {
  label: string;
  value: string;
  /** Differs from the default: drawn in the task accent. */
  changed?: boolean;
}

export interface ReviewSection {
  step: StepId;
  rows: ReviewRow[];
}

interface ReviewStepProps {
  taskKey: string;
  name: string;
  onNameChange: (name: string) => void;
  /** The name the configuration suggests (lib/experiment-name.ts). */
  suggestion: string;
  /** Dotted paths a sweep varies; each trial adds its value to the name. */
  sweepKeys?: string[];
  onExportYaml: () => void;
  onImportConfig: (
    data: Record<string, unknown>,
  ) => string | null | Promise<string | null>;
  sections: ReviewSection[];
  /** Asks for the rate of each trial; absent on a form with no such list. */
  learningRatePlan?: LearningRatePlanRequest | null;
  /** Messages of the last failed submission. */
  errors?: string[];
  /** The required data is set: Treinar can run. */
  ready: boolean;
}

const card: React.CSSProperties = {
  background: "var(--vf-panel)",
  border: "1px solid var(--vf-panel-stroke)",
  borderRadius: 14,
  padding: "16px 18px",
  minWidth: 0,
};

const kicker: React.CSSProperties = {
  fontFamily: "var(--font-mono)",
  fontSize: 10,
  letterSpacing: "0.2em",
  textTransform: "uppercase",
  color: "var(--vf-text-muted)",
};

/** The last step (ADR-156): the name, suggested and editable; one card per
 *  step with its key values; the problems of the last attempt. Treinar itself
 *  stays in the bottom bar, the only one on the page. */
export function ReviewStep({
  taskKey,
  name,
  onNameChange,
  suggestion,
  sweepKeys,
  onExportYaml,
  onImportConfig,
  sections,
  learningRatePlan,
  errors = [],
  ready,
}: ReviewStepProps) {
  const t = useT();
  const nav = useStepNav();
  const r = t.stepShell.review;
  const keys = [...new Set((sweepKeys ?? []).map(shortKey))].join(", ");

  return (
    <div style={{ display: "grid", gap: 16 }}>
      <ExperimentNameCard
        title={r.nameTitle}
        name={name}
        onNameChange={onNameChange}
        placeholder={suggestion}
        hint={keys ? `${r.nameHint} · ${r.trialNote(keys)}` : r.nameHint}
        extraAction={
          name !== suggestion ? (
            <button
              type="button"
              onClick={() => onNameChange(suggestion)}
              className="vf-step-nav"
              title={suggestion}
            >
              {r.useSuggestion}
            </button>
          ) : null
        }
        onExportYaml={onExportYaml}
        onImportConfig={onImportConfig}
      />

      <div className="vf-review-grid">
        {sections
          .filter((s) => s.rows.length > 0)
          .map((s) => (
            <div key={s.step} style={card}>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "baseline",
                  marginBottom: 10,
                }}
              >
                <span style={kicker}>{t.stepShell.steps[s.step]}</span>
                {nav && (
                  <button
                    type="button"
                    className="vf-link-btn"
                    onClick={() => nav.setStep(taskKey, s.step)}
                  >
                    {r.edit}
                  </button>
                )}
              </div>
              <dl className="vf-review-rows">
                {s.rows.map((row) => (
                  <div key={row.label} data-changed={row.changed ? "" : undefined}>
                    <dt>{row.label}</dt>
                    <dd>{row.value}</dd>
                  </div>
                ))}
              </dl>
            </div>
          ))}
      </div>

      {learningRatePlan !== undefined && (
        <LearningRatePlanCard request={learningRatePlan} />
      )}

      {errors.length > 0 && (
        <div className="vf-review-errors" role="alert">
          <div style={kicker}>{r.errorsTitle}</div>
          <ul>
            {errors.map((e) => (
              <li key={e}>{e}</li>
            ))}
          </ul>
        </div>
      )}

      <div className="vf-review-status" data-ready={ready ? "" : undefined}>
        {ready ? r.ready : r.blocked}
      </div>
    </div>
  );
}
