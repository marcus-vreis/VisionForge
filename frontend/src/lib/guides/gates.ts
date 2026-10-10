/** Gate evaluation for guided steps (ADR-115). Pure: no DOM, no React.
 *
 * The training hook keeps the previous run's events until the next submission or
 * until its results are closed. Reading "has a `start` event been seen" straight
 * off that list would open a "training started" gate the moment the guide opens
 * after a finished run. So the guide keeps a latch: it notes what was already
 * there when it opened, and counts an event only when it appears afterwards.
 */

import type { GuideFacts, GuideStep } from "./types";

/** The only part of a training event the gates read. */
interface EventLike {
  event: string;
}

export interface EventLatch {
  /** A `start` appeared after the latch was opened. */
  sawStart: boolean;
  /** The run that started then also ended. */
  sawEnd: boolean;
  /** Whether `start` / `end` were in the list at the last look. */
  hadStart: boolean;
  hadEnd: boolean;
}

const has = (events: readonly EventLike[], kind: string): boolean =>
  events.some((e) => e.event === kind);

/** The latch at the moment the guide opens: whatever is there already does not count. */
export function openLatch(events: readonly EventLike[]): EventLatch {
  return {
    sawStart: false,
    sawEnd: false,
    hadStart: has(events, "start"),
    hadEnd: has(events, "end"),
  };
}

/** Folds the current events into the latch. Returns `prev` itself when nothing
 *  changed, so a caller can tell by identity whether to update its state. */
export function advanceLatch(
  prev: EventLatch,
  events: readonly EventLike[],
): EventLatch {
  const hasStart = has(events, "start");
  const hasEnd = has(events, "end");
  const sawStart = prev.sawStart || (hasStart && !prev.hadStart);
  // An `end` only counts for a run whose `start` was also seen after opening:
  // a run already in flight when the guide opened is not the researcher's.
  const sawEnd = prev.sawEnd || (sawStart && hasEnd && !prev.hadEnd);
  if (
    sawStart === prev.sawStart &&
    sawEnd === prev.sawEnd &&
    hasStart === prev.hadStart &&
    hasEnd === prev.hadEnd
  ) {
    return prev;
  }
  return { sawStart, sawEnd, hadStart: hasStart, hadEnd: hasEnd };
}

export function guideFacts(latch: EventLatch, datasetPath: string): GuideFacts {
  return {
    datasetPath,
    trainingStarted: latch.sawStart,
    trainingEnded: latch.sawEnd,
  };
}

/** Whether the researcher may move past `step`. A step with no `waitFor` is always open. */
export function gateOpen(step: GuideStep, facts: GuideFacts): boolean {
  return step.waitFor ? step.waitFor(facts) : true;
}
