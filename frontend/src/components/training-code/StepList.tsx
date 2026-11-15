import type { CSSProperties, ReactNode } from "react";
import { useT } from "../../i18n/useT";
import { stepWords, type RunStep } from "../../lib/run-steps";

const MONO: CSSProperties = { fontFamily: "var(--font-mono)" };

/**
 * The run as steps, top to bottom, each with one plain sentence. The step (or
 * the group of steps) the stream proves is running carries the accent bar, and
 * its live figures sit under it. A click on a step shows its code.
 */
export function StepList({
  steps,
  running,
  selected,
  accent,
  patchcore,
  figures,
  onPick,
}: {
  steps: readonly RunStep[];
  running: readonly RunStep[];
  selected: RunStep;
  accent: string;
  patchcore: boolean;
  /** What the trainer reported, shown under the running step (or group). */
  figures: ReactNode;
  onPick: (step: RunStep) => void;
}) {
  const words = useT().codeTour;
  const lastRunning = running[running.length - 1];
  return (
    <ol aria-label={words.stepsLabel} style={{ listStyle: "none", margin: 0, padding: 0, display: "flex", flexDirection: "column" }}>
      {steps.map((step) => {
        const isRunning = running.includes(step);
        const isSelected = step === selected;
        const { title, text } = stepWords(words, step, patchcore);
        return (
          <li
            key={step}
            data-step={step}
            data-running={isRunning ? "true" : undefined}
            aria-current={isRunning ? "step" : undefined}
            style={{ borderLeft: `3px solid ${isRunning ? accent : "var(--vf-panel-stroke)"}` }}
          >
            <button
              type="button"
              aria-pressed={isSelected}
              onClick={() => onPick(step)}
              style={{
                display: "flex",
                flexDirection: "column",
                gap: 2,
                width: "100%",
                textAlign: "left",
                padding: "7px 10px",
                border: "none",
                borderRadius: "0 8px 8px 0",
                cursor: "pointer",
                background: isSelected ? "var(--accent-soft)" : "transparent",
                color: "var(--vf-text)",
              }}
            >
              <span style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 13.5, fontWeight: 600 }}>
                {title}
                {isRunning && (
                  <span aria-hidden="true" style={{ width: 7, height: 7, borderRadius: "50%", background: accent, flexShrink: 0 }} />
                )}
              </span>
              <span style={{ fontSize: 13, lineHeight: 1.45, color: isSelected ? "var(--vf-text)" : "var(--vf-text-dim)" }}>
                {text}
              </span>
            </button>
            {step === lastRunning && figures}
          </li>
        );
      })}
    </ol>
  );
}

/** One reported figure: a mono label and its value. */
export function Figure({ label, value }: { label: string; value: string }) {
  return (
    <span style={{ display: "inline-flex", alignItems: "baseline", gap: 5 }}>
      <span style={{ ...MONO, fontSize: 11, color: "var(--vf-text-muted)" }}>{label}</span>
      <span style={{ ...MONO, fontSize: 13, fontWeight: 600, color: "var(--vf-text)" }}>{value}</span>
    </span>
  );
}
