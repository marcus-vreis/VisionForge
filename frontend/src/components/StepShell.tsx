import { useEffect, useRef, useState, type ReactNode } from "react";
import { useT } from "../i18n/useT";
import {
  REVEAL_STEP_EVENT,
  useStepNav,
  type RevealStepDetail,
  type StepId,
} from "../lib/steps";

export interface StepDef {
  id: StepId;
  /** One line under the label in the rail: the value that matters most. */
  summary?: string;
  /** Fields of this step that differ from their defaults. */
  changes?: number;
  /** A required value of this step is empty. */
  missing?: boolean;
  /** Validation errors the server returned for fields of this step. */
  errors?: number;
  content: ReactNode;
}

/** The task configuration split into steps (ADR-156).
 *
 * Every step stays mounted and the inactive ones are only `hidden`: the
 * strategy cards listen to the bottom bar's run signal, and the dataset stats
 * fill other fields as they load, whichever step is on screen. The rail is
 * navigation, not a gate: any step opens at any time. The Treinar button in
 * the bottom bar unlocks on the Revisão step (App owns that rule).
 */
export function StepShell({
  taskKey,
  steps,
}: {
  taskKey: string;
  steps: StepDef[];
}) {
  const t = useT();
  const nav = useStepNav();
  const [localStep, setLocalStep] = useState<StepId>(steps[0]?.id ?? "data");
  const wanted = nav ? nav.stepFor(taskKey) : localStep;
  const active = steps.some((s) => s.id === wanted) ? wanted : steps[0].id;
  const setStep = (id: StepId) => {
    if (nav) nav.setStep(taskKey, id);
    else setLocalStep(id);
  };
  const setStepRef = useRef(setStep);
  useEffect(() => {
    setStepRef.current = setStep;
  });
  const topRef = useRef<HTMLDivElement>(null);

  const go = (id: StepId) => {
    setStep(id);
    // A long step leaves the next one's top off screen.
    const top = topRef.current?.getBoundingClientRect().top;
    if (top != null && top < 0) {
      topRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  };

  // The guided tour asks for the step that holds its target.
  useEffect(() => {
    const onReveal = (e: Event) => {
      const step = (e as CustomEvent<RevealStepDetail>).detail?.step;
      if (step) setStepRef.current(step);
    };
    window.addEventListener(REVEAL_STEP_EVENT, onReveal);
    return () => window.removeEventListener(REVEAL_STEP_EVENT, onReveal);
  }, []);

  // New server errors bring the first step that has one to the front.
  const totalErrors = steps.reduce((n, s) => n + (s.errors ?? 0), 0);
  const firstWithErrors = steps.find((s) => (s.errors ?? 0) > 0)?.id;
  const lastErrors = useRef(totalErrors);
  useEffect(() => {
    if (totalErrors > lastErrors.current && firstWithErrors) {
      setStepRef.current(firstWithErrors);
    }
    lastErrors.current = totalErrors;
  }, [totalErrors, firstWithErrors]);

  const index = steps.findIndex((s) => s.id === active);
  const prev = index > 0 ? steps[index - 1] : null;
  const next = index < steps.length - 1 ? steps[index + 1] : null;

  return (
    <div ref={topRef} className="vf-steps" data-active-step={active}>
      <nav className="vf-steps-rail" aria-label={t.stepShell.navLabel}>
        {steps.map((s, i) => (
          <button
            key={s.id}
            type="button"
            className="vf-step-tab"
            aria-current={s.id === active ? "step" : undefined}
            onClick={() => go(s.id)}
          >
            <span className="vf-step-num">{String(i + 1).padStart(2, "0")}</span>
            <span className="vf-step-text">
              <span className="vf-step-label">{t.stepShell.steps[s.id]}</span>
              {s.summary && <span className="vf-step-summary">{s.summary}</span>}
            </span>
            <StepChip step={s} />
          </button>
        ))}
      </nav>

      <div className="vf-steps-body">
        <header className="vf-step-head">
          <div className="vf-step-kicker">
            {t.stepShell.stepLabel(index + 1, steps.length)}
          </div>
          <h2 className="vf-step-title">{t.stepShell.steps[active]}</h2>
          <p className="vf-step-lead">{t.stepShell.leads[active]}</p>
        </header>

        {steps.map((s) => (
          <section
            key={s.id}
            data-step={s.id}
            hidden={s.id !== active}
            className="vf-step-panel"
          >
            {s.content}
          </section>
        ))}

        <footer className="vf-step-foot">
          {prev ? (
            <button type="button" className="vf-step-nav" onClick={() => go(prev.id)}>
              {t.stepShell.prev}
            </button>
          ) : (
            <span />
          )}
          {next && (
            <button
              type="button"
              className="vf-step-nav vf-step-nav-primary"
              onClick={() => go(next.id)}
            >
              {next.id === "review" ? t.stepShell.goReview : t.stepShell.next}
            </button>
          )}
        </footer>
      </div>
    </div>
  );
}

function StepChip({ step }: { step: StepDef }) {
  const t = useT();
  if ((step.errors ?? 0) > 0) {
    return (
      <span className="vf-step-chip" data-tone="error">
        {t.stepShell.status.errors(step.errors ?? 0)}
      </span>
    );
  }
  if (step.missing) {
    return (
      <span className="vf-step-chip" data-tone="missing">
        {t.stepShell.status.missing}
      </span>
    );
  }
  if (step.changes == null || step.id === "review") return null;
  if (step.changes > 0) {
    return (
      <span className="vf-step-chip" data-tone="changed">
        {t.stepShell.status.changes(step.changes)}
      </span>
    );
  }
  return <span className="vf-step-chip">{t.stepShell.status.default}</span>;
}
