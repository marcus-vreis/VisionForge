/** The generic replay of a finished run on the "Como funciona" tab (ADR-118).
 *
 * Not the run played back: the stages data -> training -> testing in slow motion,
 * one after the other, over the run's final numbers. The sequence and its clock
 * live here so they can be tested without a DOM; `use-replay.ts` only holds the
 * result in React state.
 */

import type { StoryFamily } from "./story-family";
import type { StoryStage } from "./training-story";

export interface ReplayStep {
  stage: StoryStage;
  ms: number;
}

/** About 13 s in all: training gets the longest look, it is where the flows are. */
export const REPLAY_STEPS: readonly ReplayStep[] = [
  { stage: "preparing", ms: 3000 },
  { stage: "training", ms: 6000 },
  { stage: "testing", ms: 4000 },
];

/** Only a finished run with a drawing can be replayed (PatchCore has no tab). */
export function canReplay(stage: StoryStage, family: StoryFamily): boolean {
  return stage === "done" && family !== "patchcore";
}

export interface ReplayController {
  start(): void;
  cancel(): void;
}

/** Walks `REPLAY_STEPS`, calling `onStage(stage)` for each and `onStage(null)` at the end. */
export function createReplay(
  onStage: (stage: StoryStage | null) => void,
  steps: readonly ReplayStep[] = REPLAY_STEPS,
): ReplayController {
  let timer: ReturnType<typeof setTimeout> | null = null;
  const clear = () => {
    if (timer !== null) clearTimeout(timer);
    timer = null;
  };
  const run = (i: number) => {
    if (i >= steps.length) {
      timer = null;
      onStage(null);
      return;
    }
    onStage(steps[i].stage);
    timer = setTimeout(() => run(i + 1), steps[i].ms);
  };
  return {
    // Restarting mid-way begins again instead of stacking a second clock.
    start() {
      clear();
      run(0);
    },
    cancel() {
      clear();
      onStage(null);
    },
  };
}
