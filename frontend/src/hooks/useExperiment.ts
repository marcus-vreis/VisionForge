import { useCallback, useEffect, useRef, useState } from "react";
import {
  ApiError,
  fetchResult,
  fetchStatus,
  runExperiment,
} from "../api/client";
import type { RunResult, RunStatus } from "../types/run";

interface ExperimentState {
  status: RunStatus;
  result: RunResult | null;
  error: string | null;
  validationErrors: ValidationError[];
  submit: (config: Record<string, unknown>) => Promise<void>;
  reset: () => void;
}

export interface ValidationError {
  field: string[];
  message: string;
}

export function useExperiment(): ExperimentState {
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
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const startPolling = useCallback(
    (_runId: string) => {
      stopPolling();
      pollRef.current = setInterval(async () => {
        try {
          const s = await fetchStatus();
          setStatus(s);

          if (s.status === "completed" && s.run_id) {
            stopPolling();
            const r = await fetchResult(s.run_id);
            setResult(r);
          } else if (s.status === "failed") {
            stopPolling();
            setError(s.error ?? "Experiment failed.");
          }
        } catch {
          stopPolling();
          setError("Lost connection to server.");
        }
      }, 2000);
    },
    [stopPolling],
  );

  useEffect(() => stopPolling, [stopPolling]);

  const submit = useCallback(
    async (config: Record<string, unknown>) => {
      setError(null);
      setResult(null);
      setValidationErrors([]);

      try {
        const res = await runExperiment(config);
        setStatus({ status: "running", run_id: res.run_id, error: null });
        startPolling(res.run_id);
      } catch (e) {
        if (e instanceof ApiError && e.status === 422) {
          try {
            const body = JSON.parse(e.message);
            if (Array.isArray(body)) {
              setValidationErrors(
                body.map((err: { loc: string[]; msg: string }) => ({
                  field: err.loc.filter((l) => l !== "body"),
                  message: err.msg,
                })),
              );
              return;
            }
          } catch {
            // not JSON, fall through
          }
          setError(e.message);
        } else if (e instanceof ApiError && e.status === 409) {
          setError("An experiment is already running.");
        } else if (e instanceof Error) {
          setError(e.message);
        }
      }
    },
    [startPolling],
  );

  const reset = useCallback(() => {
    stopPolling();
    setStatus({ status: "idle", run_id: null, error: null });
    setResult(null);
    setError(null);
    setValidationErrors([]);
  }, [stopPolling]);

  return { status, result, error, validationErrors, submit, reset };
}
