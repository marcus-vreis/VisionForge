import type { AnalysisState } from "../lib/analysis-job";
import { scanPercent } from "../lib/dataset-explorer";
import { useT } from "../i18n/useT";
import { ActionButton } from "./ui/ActionButton";
import { card, danger, ghostButton, mono, stepTitle, text, warn } from "./datasetExplorerStyles";

/** The words a card needs; each analysis brings its own. */
export interface AnalysisWords {
  title: string;
  help: string;
  start: string;
  progress: (done: number, total: number) => string;
}

/** Title, explanation, start/cancel, progress, errors and the kept-result note. */
export function AnalysisJobCard<R>({
  words,
  state,
  ready,
}: {
  words: AnalysisWords;
  state: AnalysisState<R>;
  ready: boolean;
}) {
  const t = useT().analysisJob;
  const { job, running, cancelling, error, saved } = state;
  const done = job ? scanPercent(job.done, job.total) : 0;
  return (
    <div style={card}>
      <h3 style={{ ...stepTitle, margin: 0 }}>{words.title}</h3>
      <div style={text}>{words.help}</div>
      <div>
        {running ? (
          <button type="button" style={ghostButton} onClick={() => void state.cancel()} disabled={cancelling}>
            {cancelling ? t.cancelling : t.cancel}
          </button>
        ) : (
          <ActionButton onClick={() => void state.start()} disabled={!ready}>
            {words.start}
          </ActionButton>
        )}
      </div>
      {running && job && (
        <div>
          <div style={mono}>{words.progress(job.done, job.total)}</div>
          <div
            role="progressbar"
            aria-valuenow={done}
            aria-valuemin={0}
            aria-valuemax={100}
            style={{ height: 6, borderRadius: 3, background: "rgb(var(--lt-ink, 255 255 255) / 0.08)" }}
          >
            <div style={{ height: "100%", width: `${done}%`, borderRadius: 3, background: "var(--accent-vf)" }} />
          </div>
        </div>
      )}
      {error && <div style={danger}>{t.failed(error)}</div>}
      {job?.state === "failed" && job.error && <div style={danger}>{t.failed(job.error)}</div>}
      {!job && saved?.report && (
        <div style={saved.stale ? warn : text} role="status">
          {t.savedAt(saved.saved_at ? new Date(saved.saved_at).toLocaleString() : "")}
          {saved.stale && ` ${t.savedStale}`}
        </div>
      )}
    </div>
  );
}
