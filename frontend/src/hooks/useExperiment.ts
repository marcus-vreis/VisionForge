import { useCallback, useEffect, useRef, useState } from "react";
import {
  ApiError,
  fetchQueue,
  fetchResult,
  fetchStatus,
  runAnomaly,
  runDetection,
  runExperiment,
  runRegression,
  runSegmentation,
} from "../api/client";
import { useT } from "../i18n/useT";
import type { Dict } from "../i18n/pt";
import type {
  RunResponse,
  RunResult,
  RunStatus,
  TrainingEvent,
} from "../types/run";

interface ExperimentState {
  status: RunStatus;
  result: RunResult | null;
  error: string | null;
  validationErrors: ValidationError[];
  progressEvents: TrainingEvent[];
  submit: (
    config: Record<string, unknown>,
    opts?: {
      detection?: boolean;
      regression?: boolean;
      segmentation?: boolean;
      anomaly?: boolean;
      /** Custom submitter (e.g. comparison/sweep endpoints); overrides the
       *  task-based dispatch when provided. */
      run?: (config: Record<string, unknown>) => Promise<RunResponse>;
    },
  ) => Promise<void>;
  reset: () => void;
}

export interface ValidationError {
  field: string[];
  message: string;
}

/** Preprocessing filter ids whose Pydantic errors should appear with a
 * human-friendly name in field path summaries. These are the filters' own
 * technical names, which read the same in every language, so they stay here
 * rather than in the dictionaries. */
const PREPROCESS_KIND_LABELS: Record<string, string> = {
  gaussian_blur: "Gaussian blur",
  median_blur: "Median blur",
  unsharp: "Unsharp mask",
  edges: "Edges",
  emboss: "Emboss",
  grayscale: "Grayscale",
  equalize: "Equalize",
  autocontrast: "Autocontrast",
  wavelet: "Wavelet",
};

/** Build a user-readable path like "Treinamento › Learning Rate", with the
 * section and field names in the language of `t` (`const t = useT()`).
 *
 * Numeric segments (Pydantic list index) become "#N" so the user can tell
 * which filter slot in the pipeline failed validation. A known filter kind
 * (passed alongside via the special "kind=foo" pseudo-segment) gets its
 * human label inserted next to the index.
 */
export function humanizeFieldPath(t: Dict, loc: (string | number)[]): string {
  const sections: Record<string, string> = t.experiment.sections;
  const fields: Record<string, string> = t.experiment.fields;
  return loc
    .filter((p) => p !== "body")
    .map((p) => {
      if (typeof p === "number") {
        // List index inside steps[] — show as "#N" so the user can tell
        // which filter slot in the pipeline failed validation.
        return `#${p + 1}`;
      }
      const k = String(p);
      const known = PREPROCESS_KIND_LABELS[k];
      if (known) return known;
      return sections[k] ?? fields[k] ?? k;
    })
    .filter((s) => s !== "")
    .join(" › ");
}

export function useExperiment(): ExperimentState {
  const t = useT();
  // The polling timer and the submit callback outlive the render that made them,
  // so they word their messages from this ref and not from the dictionary they
  // closed over: a language switched mid-run still applies to what comes next.
  const tRef = useRef(t);
  useEffect(() => {
    tRef.current = t;
  }, [t]);
  const [status, setStatus] = useState<RunStatus>({
    status: "idle",
    run_id: null,
    error: null,
  });
  const [result, setResult] = useState<RunResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [validationErrors, setValidationErrors] = useState<ValidationError[]>(
    [],
  );
  const [progressEvents, setProgressEvents] = useState<TrainingEvent[]>([]);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const esRef = useRef<EventSource | null>(null);
  // Which run this hook is following, and whether it has been seen executing.
  // /experiment/status describes whatever is *active*, so with a queue (ADR-075)
  // a submission that is still waiting would otherwise read another job's
  // completion as its own and fetch the wrong result.
  const myRunRef = useRef<string | null>(null);
  const startedRef = useRef(false);

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const closeEventSource = useCallback(() => {
    if (esRef.current) {
      esRef.current.close();
      esRef.current = null;
    }
  }, []);

  const openEventSource = useCallback(() => {
    closeEventSource();
    const es = new EventSource("/api/experiment/events");
    esRef.current = es;

    es.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data) as TrainingEvent;
        // Ignore the empty sentinel emitted when no run is active.
        if (!data.event) return;
        setProgressEvents((prev) => [...prev, data]);
        if (data.event === "end") {
          closeEventSource();
        }
      } catch {
        // Malformed frame — skip.
      }
    };

    es.onerror = () => {
      // SSE failed or server closed; fall back to the existing 2s polling.
      closeEventSource();
    };
  }, [closeEventSource]);

  const collectResult = useCallback(
    async (runId: string) => {
      stopPolling();
      try {
        setResult(await fetchResult(runId));
        setStatus({ status: "completed", run_id: runId, error: null });
      } catch (e) {
        const msg =
          e instanceof ApiError
            ? e.message
            : tRef.current.experiment.resultFetchFailed;
        setError(msg);
        setStatus({ status: "failed", run_id: runId, error: msg });
      }
    },
    [stopPolling],
  );

  const startPolling = useCallback(
    () => {
      stopPolling();
      pollRef.current = setInterval(async () => {
        try {
          const s = await fetchStatus();
          const mine = myRunRef.current;

          if (mine !== null && s.run_id !== mine) {
            if (startedRef.current) {
              // Our run held the slot and no longer does: it finished and the
              // next queued job took over. Its report is still fetchable by id.
              await collectResult(mine);
              return;
            }
            // Still waiting behind someone else — show where in line.
            try {
              const q = await fetchQueue();
              const index = q.pending.findIndex((p) => p.run_id === mine);
              setStatus({
                status: "queued",
                run_id: mine,
                error: null,
                queued: q.pending.length,
                position: index >= 0 ? index + 1 : undefined,
              });
            } catch {
              // Queue endpoint hiccup — keep waiting rather than failing the run.
            }
            return;
          }

          if (mine !== null && s.status === "running") {
            if (!startedRef.current) {
              startedRef.current = true;
              openEventSource(); // it is our turn: attach to the live stream
            }
          }
          setStatus(s);

          if (s.status === "completed" && s.run_id) {
            await collectResult(s.run_id);
          } else if (s.status === "failed") {
            stopPolling();
            setError(
              s.error ?? tRef.current.experiment.failedNoDetail,
            );
          }
        } catch (e) {
          stopPolling();
          const msg =
            e instanceof ApiError
              ? e.message
              : tRef.current.experiment.connectionLost;
          setError(msg);
          setStatus((prev) => ({
            status: "failed",
            run_id: prev.run_id,
            error: msg,
          }));
        }
      }, 2000);
    },
    [stopPolling, collectResult, openEventSource],
  );

  useEffect(() => () => {
    stopPolling();
    closeEventSource();
  }, [stopPolling, closeEventSource]);

  const submit = useCallback(
    async (
      config: Record<string, unknown>,
      opts?: {
      detection?: boolean;
      regression?: boolean;
      segmentation?: boolean;
      anomaly?: boolean;
      /** Custom submitter (e.g. comparison/sweep endpoints); overrides the
       *  task-based dispatch when provided. */
      run?: (config: Record<string, unknown>) => Promise<RunResponse>;
    },
    ) => {
      setError(null);
      setResult(null);
      setValidationErrors([]);
      setProgressEvents([]);

      try {
        const res = await (opts?.run
          ? opts.run(config)
          : opts?.detection
            ? runDetection(config)
            : opts?.regression
              ? runRegression(config)
              : opts?.segmentation
                ? runSegmentation(config)
                : opts?.anomaly
                  ? runAnomaly(config)
                  : runExperiment(config));
        myRunRef.current = res.run_id;
        startedRef.current = res.status === "running";
        if (res.status === "queued") {
          // No live stream yet — polling promotes this to "running" and attaches
          // the EventSource when the queue reaches this job.
          setStatus({ status: "queued", run_id: res.run_id, error: null });
        } else {
          setStatus({ status: "running", run_id: res.run_id, error: null });
          openEventSource();
        }
        startPolling();
      } catch (e) {
        if (e instanceof ApiError) {
          if (e.status === 422 && e.validationErrors) {
            setValidationErrors(
              e.validationErrors.map((err) => ({
                field: err.loc
                  .filter((l) => l !== "body")
                  .map((l) => String(l)),
                message: err.msg,
              })),
            );
            setError(
              tRef.current.experiment.validationFailed(e.validationErrors.length),
            );
            return;
          }
          if (e.status === 409) {
            // Kept for older servers: since ADR-075 a busy server queues the
            // submission instead of refusing it.
            setError(tRef.current.experiment.alreadyRunning);
            return;
          }
          if (e.status === 0) {
            setError(e.message);
            return;
          }
          setError(e.message);
          return;
        }
        if (e instanceof Error) {
          setError(tRef.current.experiment.unexpected(e.message));
          return;
        }
        setError(tRef.current.experiment.unknown);
      }
    },
    [startPolling, openEventSource],
  );

  const reset = useCallback(() => {
    stopPolling();
    closeEventSource();
    myRunRef.current = null;
    startedRef.current = false;
    setStatus({ status: "idle", run_id: null, error: null });
    setResult(null);
    setError(null);
    setValidationErrors([]);
    setProgressEvents([]);
  }, [stopPolling, closeEventSource]);

  return { status, result, error, validationErrors, progressEvents, submit, reset };
}
