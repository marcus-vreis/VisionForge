/** The replay's state: the stage it forces on the tab, or `null` for the real one (ADR-118). */

import { useCallback, useEffect, useRef, useState } from "react";
import { createReplay, type ReplayController } from "../../lib/training-replay";
import type { StoryStage } from "../../lib/training-story";

export function useReplay(active: boolean): {
  stage: StoryStage | null;
  start: () => void;
} {
  const [stage, setStage] = useState<StoryStage | null>(null);
  const controller = useRef<ReplayController | null>(null);

  useEffect(() => {
    const replay = createReplay(setStage);
    controller.current = replay;
    return () => {
      replay.cancel();
      controller.current = null;
    };
  }, []);

  // A new run (or anything that stops the story being "done") ends the replay.
  useEffect(() => {
    if (!active) controller.current?.cancel();
  }, [active]);

  const start = useCallback(() => controller.current?.start(), []);
  return { stage, start };
}
