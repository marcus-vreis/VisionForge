import { createSample } from "./guides/first-training";
import type { GuideTask } from "./guides/tasks";
import type { SampleSetup } from "./guides/types";

/** What `createSample` failed with, as the sentence the panel shows. */
export async function runSample(
  task: GuideTask,
  apply: (path: string, setup: SampleSetup) => void,
  failed: (reason: string) => string,
): Promise<string | null> {
  try {
    const made = await createSample(task);
    apply(made.path, made.setup);
    return null;
  } catch (e) {
    return failed(e instanceof Error ? e.message : String(e));
  }
}
