import { useState } from "react";
import { useT } from "../i18n/useT";
import { createSample } from "../lib/guides/first-training";
import type { GuideTask } from "../lib/guides/tasks";
import type { SampleSetup } from "../lib/guides/types";

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

/** Small button under a task's folder row: creates the sample dataset (or reuses
 *  the one already there) and hands its path to the panel, as the guide does. */
export function SampleDatasetButton({
  task,
  onApply,
}: {
  task: GuideTask;
  onApply: (path: string, setup: SampleSetup) => void;
}) {
  const t = useT();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onClick = async () => {
    setBusy(true);
    try {
      setError(await runSample(task, onApply, t.sampleDataset.failed));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div style={{ gridColumn: "1 / -1", marginTop: -6 }}>
      <button
        type="button"
        data-sample-task={task}
        onClick={() => void onClick()}
        disabled={busy}
        title={t.sampleDataset.title}
        style={{
          padding: 0,
          background: "none",
          border: "none",
          color: "var(--vf-text-dim)",
          fontFamily: "var(--font-mono)",
          fontSize: 11,
          textDecoration: "underline",
          cursor: busy ? "default" : "pointer",
        }}
      >
        {busy ? "…" : t.sampleDataset.button}
      </button>
      {error && (
        <div
          role="alert"
          style={{
            marginTop: 4,
            fontFamily: "var(--font-mono)",
            fontSize: 11,
            color: "oklch(0.85 0.14 22)",
          }}
        >
          {error}
        </div>
      )}
    </div>
  );
}
