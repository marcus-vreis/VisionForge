/** A dataset analysis that runs as an explorer job and keeps its last result
 *  (ADR-234, ADR-235): start it, follow its progress, cancel it, and show the
 *  kept result of the same request when nothing is running. */
import { useCallback, useEffect, useState } from "react";
import { isScanFinished, type ScanState } from "./dataset-explorer";

export interface AnalysisJob<R> {
  job_id: string;
  state: ScanState;
  done: number;
  total: number;
  result: R | null;
  error: string | null;
}

export interface SavedAnalysis<R> {
  report: R | null;
  saved_at: string | null;
  stale: boolean;
}

export interface AnalysisApi<Q, R> {
  start: (req: Q) => Promise<AnalysisJob<R>>;
  status: (jobId: string) => Promise<AnalysisJob<R>>;
  saved: (req: Q) => Promise<SavedAnalysis<R>>;
  cancel: (jobId: string) => Promise<void>;
}

export interface AnalysisState<R> {
  job: AnalysisJob<R> | null;
  running: boolean;
  cancelling: boolean;
  error: string | null;
  /** The kept result for this request, shown while no job of this page answers. */
  saved: SavedAnalysis<R> | null;
  /** What to show: the job's result, else the kept one when nothing runs. */
  shown: R | null;
  start: () => Promise<void>;
  cancel: () => Promise<void>;
}

const POLL_MS = 600;

function message(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

export function useAnalysisJob<Q, R>(api: AnalysisApi<Q, R>, request: Q, ready: boolean): AnalysisState<R> {
  const [job, setJob] = useState<AnalysisJob<R> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [cancelling, setCancelling] = useState(false);
  const [kept, setKept] = useState<{ key: string; saved: SavedAnalysis<R> } | null>(null);
  // The request object may be rebuilt on every render; what it names is what counts.
  const key = JSON.stringify(request);
  const saved = kept?.key === key ? kept.saved : null;
  const jobId = job?.job_id ?? null;
  const running = job !== null && !isScanFinished(job.state);

  useEffect(() => {
    if (!ready) return;
    let live = true;
    api
      .saved(request)
      .then((s) => {
        if (live) setKept({ key, saved: s });
      })
      // No kept result is not an error worth a message.
      .catch(() => undefined);
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- keyed by the request
  }, [key, ready]);

  useEffect(() => {
    if (!jobId || !running) return;
    const timer = setInterval(() => {
      api
        .status(jobId)
        .then((next) => {
          setJob(next);
          if (isScanFinished(next.state)) setCancelling(false);
        })
        .catch((e: unknown) => {
          setError(message(e));
          setCancelling(false);
        });
    }, POLL_MS);
    return () => clearInterval(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps -- the api object is static
  }, [jobId, running]);

  const start = useCallback(async () => {
    setError(null);
    setJob(null);
    try {
      setJob(await api.start(request));
    } catch (e) {
      setError(message(e));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- keyed by the request
  }, [key]);

  const cancel = async () => {
    if (!jobId) return;
    setCancelling(true);
    try {
      await api.cancel(jobId);
    } catch (e) {
      setError(message(e));
      setCancelling(false);
    }
  };

  const shown = job?.result ?? (running ? null : (saved?.report ?? null));
  return { job, running, cancelling, error, saved, shown, start, cancel };
}
