/** The run queues of the server, one per resource (ADR-163 phase 4).
 *
 * `local` is the line ADR-075 introduced: one job at a time on this machine,
 * GPU or CPU. `colab:<e-mail>` is one per Google account, which the profiles
 * connected to that account share. A local job and a Colab job run at once;
 * two jobs of one queue never do.
 *
 * `/api/queue` keeps `active`/`pending` for the local queue and adds `lanes`
 * with all of them. A server older than phase 4 sends no `lanes`: its one
 * queue is the local one.
 */
import type { QueuedJobInfo, QueueLane, QueueSnapshot } from "../types/run";

export const LOCAL_LANE = "local";
const COLAB_PREFIX = "colab:";
// A Colab lane whose account the CLI could not name is keyed by the profile.
const PROFILE_PREFIX = "profile:";

/** Every queue in the snapshot, the local one first and the Colab ones by key. */
export function queueLanes(snap: QueueSnapshot): QueueLane[] {
  const lanes = snap.lanes ?? [
    { lane: LOCAL_LANE, active: snap.active, pending: snap.pending },
  ];
  const local = lanes.filter((l) => l.lane === LOCAL_LANE);
  const others = lanes
    .filter((l) => l.lane !== LOCAL_LANE)
    .sort((a, b) => a.lane.localeCompare(b.lane));
  return [...local, ...others];
}

/** How many jobs wait across all queues, and whether any is running: what the
 *  bottom bar's badge and its queue button follow. */
export function queueTotals(snap: QueueSnapshot): {
  pending: number;
  running: boolean;
} {
  const lanes = queueLanes(snap);
  return {
    pending: lanes.reduce((n, l) => n + l.pending.length, 0),
    running: lanes.some((l) => l.active !== null),
  };
}

/** Where a job is: its queue, whether it runs, and its place in that queue's
 *  line (1-based) while it waits. Null when no queue holds it. */
export function findQueuedJob(
  snap: QueueSnapshot,
  runId: string,
): {
  lane: string;
  job: QueuedJobInfo;
  running: boolean;
  position?: number;
  waiting: number;
} | null {
  for (const l of queueLanes(snap)) {
    if (l.active?.run_id === runId) {
      return { lane: l.lane, job: l.active, running: true, waiting: l.pending.length };
    }
    const index = l.pending.findIndex((p) => p.run_id === runId);
    if (index >= 0) {
      return {
        lane: l.lane,
        job: l.pending[index],
        running: false,
        position: index + 1,
        waiting: l.pending.length,
      };
    }
  }
  return null;
}

/** The Colab queue of the connected account (the lane key the server uses). */
export function colabLane(account: string): string {
  return `${COLAB_PREFIX}${account.toLowerCase()}`;
}

/** The queues to show: the server's, plus an empty one for the connected Colab
 *  account until its first job creates it on the server. */
export function withAccountLane(
  lanes: QueueLane[],
  account: string | null,
): QueueLane[] {
  if (account === null) return lanes;
  const key = colabLane(account);
  if (lanes.some((l) => l.lane === key)) return lanes;
  return [...lanes, { lane: key, active: null, pending: [] }];
}

export function isColabLane(lane: string | undefined): boolean {
  return lane !== undefined && lane.startsWith(COLAB_PREFIX);
}

/** The Google account a Colab queue belongs to (or the profile, when the CLI
 *  could not name the account); null for the local queue. */
export function laneAccount(lane: string): string | null {
  if (!isColabLane(lane)) return null;
  const rest = lane.slice(COLAB_PREFIX.length);
  return rest.startsWith(PROFILE_PREFIX) ? rest.slice(PROFILE_PREFIX.length) : rest;
}
