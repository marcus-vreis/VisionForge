import { createContext, useContext } from "react";

/** The six steps every task's configuration is split into (ADR-138). The order
 *  is the suggested sequence, but any step opens at any time. */
export const STEP_IDS = [
  "data",
  "strategy",
  "model",
  "training",
  "image",
  "review",
] as const;

export type StepId = (typeof STEP_IDS)[number];

/** Which step each task shows, kept in App so switching task tabs (which
 *  unmounts the panel) does not send the researcher back to step 1. */
export interface StepNav {
  stepFor: (taskKey: string) => StepId;
  setStep: (taskKey: string, step: StepId) => void;
}

export const StepNavContext = createContext<StepNav | null>(null);

/** Falls back to local state semantics when no provider is mounted (tests). */
export function useStepNav(): StepNav | null {
  return useContext(StepNavContext);
}

/** The event the guided tour sends when its target sits in a hidden step. */
export const REVEAL_STEP_EVENT = "vf:reveal-step";

export interface RevealStepDetail {
  step: StepId;
}

export function isStepId(value: string | null | undefined): value is StepId {
  return value != null && (STEP_IDS as readonly string[]).includes(value);
}

/** Shows the step that holds `el`, or `fallback`. Returns whether a step was
 *  asked to open. */
export function revealStepFor(
  el: Element | null,
  fallback?: StepId,
): boolean {
  const holder = el?.closest<HTMLElement>("[data-step]");
  const step = holder?.dataset.step ?? fallback;
  if (!isStepId(step)) return false;
  window.dispatchEvent(
    new CustomEvent<RevealStepDetail>(REVEAL_STEP_EVENT, { detail: { step } }),
  );
  return true;
}

/** Number of dotted paths whose value in `current` differs from `defaults`. */
export function countChanged(
  current: unknown,
  defaults: unknown,
  paths: readonly string[],
): number {
  const at = (obj: unknown, path: string): unknown =>
    path.split(".").reduce<unknown>(
      (acc, key) =>
        acc != null && typeof acc === "object"
          ? (acc as Record<string, unknown>)[key]
          : undefined,
      obj,
    );
  return paths.filter(
    (p) => JSON.stringify(at(current, p)) !== JSON.stringify(at(defaults, p)),
  ).length;
}

/** Server validation errors (`field` is the pydantic loc) whose path starts
 *  with one of `prefixes`, e.g. ["data", "model.num_classes"]. */
export function errorsUnder(
  errors: ReadonlyArray<{ field: string[] }>,
  prefixes: readonly string[],
): number {
  return errors.filter((e) => {
    const path = e.field.filter((p) => p !== "body").join(".");
    return prefixes.some((p) => path === p || path.startsWith(`${p}.`));
  }).length;
}
