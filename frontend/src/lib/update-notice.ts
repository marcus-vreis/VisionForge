/** When the corner of the header offers an update, and how the page waits for it
 *  (ADR-120). Pure functions; the component holds the state. */

import type { UpdateStatus } from "../api/client";

/** The key under which "skip this version" is remembered in this browser. */
export const SKIP_KEY = "vf.update.skip";

/** The GitHub release page of a version: its notes are that version's CHANGELOG. */
export function releaseUrl(version: string): string {
  return `https://github.com/marcus-vreis/VisionForge/releases/tag/v${version}`;
}

/** What the corner shows. */
export type NoticeMode =
  | { kind: "hidden" }
  /** A newer release, and the button can install it. */
  | { kind: "offer"; latest: string }
  /** A newer release the button cannot install; the reason and the command. */
  | { kind: "manual"; latest: string; reason: NonNullable<UpdateStatus["blocked"]>; command: string }
  /** The last update to this very version failed; the old one is running. */
  | { kind: "failed"; target: string; log: string };

export function noticeMode(status: UpdateStatus | null, skipped: string | null): NoticeMode {
  if (status === null) return { kind: "hidden" };
  const attempt = status.last_attempt;
  // A failed attempt at the version still on offer is what the reader needs to
  // know first: pressing the button again would fail the same way.
  if (attempt && !attempt.ok && status.available && attempt.target === status.latest) {
    return { kind: "failed", target: attempt.target, log: attempt.log };
  }
  if (!status.available || status.latest === null) return { kind: "hidden" };
  if (skipped === status.latest) return { kind: "hidden" };
  if (status.can_apply) return { kind: "offer", latest: status.latest };
  return {
    kind: "manual",
    latest: status.latest,
    reason: status.blocked ?? "no_installer",
    command: status.command ?? "",
  };
}

export function readSkipped(): string | null {
  try {
    return localStorage.getItem(SKIP_KEY);
  } catch {
    return null;
  }
}

export function writeSkipped(version: string): void {
  try {
    localStorage.setItem(SKIP_KEY, version);
  } catch {
    // Private mode or blocked storage: the notice simply comes back next time.
  }
}

/** How waiting for the new version ended. */
export type WaitOutcome = "updated" | "back_old" | "timeout";

/** Poll until the server answers with `target`.
 *
 * The old server goes away first (no answer), the updater installs, and a server
 * comes back. With `target`, the update worked. With another version after it had
 * gone away, the install failed and the updater relaunched the old one: stop
 * there, so the page can show the recorded failure instead of waiting on. An
 * answer with the old version before it ever went away is the old server still
 * shutting down; keep waiting. */
export async function waitForVersion(
  target: string,
  fetchVersion: () => Promise<string | null>,
  {
    timeoutMs = 300_000,
    intervalMs = 2_000,
    now = () => Date.now(),
    sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms)),
  }: {
    timeoutMs?: number;
    intervalMs?: number;
    now?: () => number;
    sleep?: (ms: number) => Promise<void>;
  } = {},
): Promise<WaitOutcome> {
  const deadline = now() + timeoutMs;
  let wentDown = false;
  while (now() < deadline) {
    await sleep(intervalMs);
    const version = await fetchVersion();
    if (version === target) return "updated";
    if (version === null) wentDown = true;
    else if (wentDown) return "back_old";
  }
  return "timeout";
}
