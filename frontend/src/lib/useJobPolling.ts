import { useEffect } from "react";
import { isJobFinished, type IncorporateJob } from "./dataset-incorporate";

export const errorMessage = (e: unknown) => (e instanceof Error ? e.message : String(e));

/** Polls a running derived-dataset job (incorporate or merge) every 500 ms until it
 *  leaves "running". `poll` must be a stable function. */
export function useJobPolling<R>(
  job: IncorporateJob<R> | null,
  poll: (jobId: string) => Promise<IncorporateJob<R>>,
  set: (j: IncorporateJob<R>) => void,
  fail: (m: string) => void,
) {
  const jobId = job?.job_id ?? null;
  const running = job !== null && !isJobFinished(job.state);
  useEffect(() => {
    if (!jobId || !running) return;
    const timer = setInterval(() => {
      poll(jobId).then(set).catch((e: unknown) => fail(errorMessage(e)));
    }, 500);
    return () => clearInterval(timer);
  }, [jobId, running, poll, set, fail]);
}
