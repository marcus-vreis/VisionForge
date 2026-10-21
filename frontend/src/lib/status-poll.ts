import { ApiError } from "../api/client";

export const POLL_MS = 2000;

/** Waits before each retry while the server does not answer: 30 s in all, then
 *  the connection counts as lost (ADR-133). */
export const RETRY_DELAYS_MS = [2000, 4000, 8000, 8000, 8000];

/** How often a lost connection is tried again, so the watch resumes by itself. */
export const LOST_RETRY_MS = 10_000;

export type ConnectionState = "reconnecting" | "lost";

export interface PollHandlers {
  /** One poll; throws what the request threw. */
  tick: () => Promise<void>;
  /** The connection changed; `null` when the server answers again. */
  onConnection: (state: ConnectionState | null) => void;
  /** The server answered with an error: the watch ends. */
  onError: (e: unknown) => void;
}

/** True when the request never got an answer (`request()` reports it as status 0). */
export function isNetworkError(e: unknown): boolean {
  return !(e instanceof ApiError) || e.status === 0;
}

/** Poll the run's status until stopped, riding out network failures; returns the stop function.
 *
 * A failure the server answered (an HTTP error) ends the watch. One where it did
 * not answer is retried with backoff: the training runs on the server whether or
 * not this page can reach it, so dropping the watch would report a failure that
 * did not happen.
 */
export function startStatusPoll(h: PollHandlers): () => void {
  let stopped = false;
  let failures = 0;
  let timer: ReturnType<typeof setTimeout> | undefined;

  const schedule = (ms: number) => {
    timer = setTimeout(() => void run(), ms);
  };

  const run = async () => {
    try {
      await h.tick();
      // The tick may have ended the watch itself (the run finished).
      if (stopped) return;
      if (failures > 0) {
        failures = 0;
        h.onConnection(null);
      }
      schedule(POLL_MS);
    } catch (e) {
      if (stopped) return;
      if (!isNetworkError(e)) {
        stopped = true;
        h.onError(e);
        return;
      }
      failures += 1;
      const lost = failures > RETRY_DELAYS_MS.length;
      h.onConnection(lost ? "lost" : "reconnecting");
      schedule(lost ? LOST_RETRY_MS : RETRY_DELAYS_MS[failures - 1]);
    }
  };

  schedule(POLL_MS);
  return () => {
    stopped = true;
    clearTimeout(timer);
  };
}
