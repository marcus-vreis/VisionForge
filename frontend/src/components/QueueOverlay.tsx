import { useCallback, useEffect, useState } from "react";

import { ApiError, cancelQueuedRun, fetchColabStatus, fetchQueue } from "../api/client";
import type { Dict } from "../i18n/pt";
import { useT } from "../i18n/useT";
import { strategyLabel, taskLabel, waitedFor } from "../lib/queue-format";
import { jobProfileLabel, queueShowsProfiles } from "../lib/profile";
import {
  laneAccount,
  LOCAL_LANE,
  queueLanes,
  queueTotals,
  withAccountLane,
} from "../lib/queue-lanes";
import { stopMode } from "../lib/run-control";
import type { QueuedJobInfo, QueueLane } from "../types/run";

interface QueueOverlayProps {
  open: boolean;
  onClose: () => void;
  /** Reports the pending count up so the bottom-bar badge cannot go stale —
   *  cancelling here changes a number this component does not own. */
  onCountChange?: (pending: number) => void;
  /** Same for whether a job is running: the bottom bar keeps the queue button
   *  up while one is, so it has to hear when that stops. */
  onRunningChange?: (running: boolean) => void;
  /** Opens the training sheet on the running job (ADR-159). */
  onWatch?: (job: QueuedJobInfo) => void;
}

/** The run queue as a surface of its own (ADR-075).
 *
 * The training overlay shows the job you just submitted; this shows the whole
 * line, which is what you want after queueing an evening's work and coming back
 * to it. A pending job is removed at once; the running one is stopped at its
 * next epoch boundary and keeps what it has saved (ADR-088).
 *
 * One section per queue (ADR-163 phase 4): this machine's and the Colab one of
 * each Google account, which run side by side. Every button acts on its own
 * row's job only.
 */
export function QueueOverlay({
  open,
  onClose,
  onCountChange,
  onRunningChange,
  onWatch,
}: QueueOverlayProps) {
  const t = useT();
  const [lanes, setLanes] = useState<QueueLane[]>([]);
  // The connected Colab account: its queue is shown even before it holds a job.
  const [colabAccount, setColabAccount] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const snap = await fetchQueue();
      const totals = queueTotals(snap);
      setLanes(queueLanes(snap));
      onCountChange?.(totals.pending);
      onRunningChange?.(totals.running);
      setError(null);
    } catch (e) {
      setError(
        e instanceof ApiError ? e.message : t.queueOverlay.readFailed,
      );
    }
  }, [onCountChange, onRunningChange, t]);

  useEffect(() => {
    if (!open) return;
    let alive = true;
    fetchColabStatus()
      .then((c) => {
        if (alive) setColabAccount(c.logged_in ? c.account : null);
      })
      .catch(() => {
        // No Colab connection to show: the queues the server reports stand alone.
      });
    return () => {
      alive = false;
    };
  }, [open]);

  // Poll while open: the queue advances on its own as jobs finish.
  useEffect(() => {
    if (!open) return;
    void refresh();
    const id = setInterval(() => void refresh(), 3000);
    return () => clearInterval(id);
  }, [open, refresh]);

  const cancel = async (runId: string) => {
    setBusyId(runId);
    try {
      await cancelQueuedRun(runId);
      await refresh();
    } catch (e) {
      setError(
        e instanceof ApiError
          ? // 409: the job is running and does not accept a stop. The server's
            // text is Portuguese; the interface words it in the reader's language.
            e.status === 409
            ? t.queueOverlay.stopRefused
            : e.message
          : t.queueOverlay.cancelFailed,
      );
    } finally {
      setBusyId(null);
    }
  };

  if (!open) return null;

  const shown = withAccountLane(lanes, colabAccount);
  const jobs = shown.flatMap((l) => (l.active ? [l.active, ...l.pending] : l.pending));
  // Only a server with more than the default profile has anyone to tell apart.
  const showProfile = queueShowsProfiles(jobs);
  // A lone local queue reads as before; with a Colab one, each gets a heading.
  const showLanes = shown.length > 1;

  return (
    <div
      onClick={onClose}
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 100,
        background: "rgb(var(--lt-surface, 4 5 7) / 0.72)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: 24,
        animation: "overlayIn 220ms ease forwards",
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          width: "min(680px, 100%)",
          maxHeight: "85vh",
          background: "rgb(var(--lt-surface, 12 14 18) / 0.95)",
          border: "1px solid var(--vf-panel-stroke)",
          borderRadius: 18,
          boxShadow: "0 50px 120px rgb(var(--lt-shade, 0 0 0) / 0.7)",
          overflow: "hidden",
          display: "flex",
          flexDirection: "column",
          animation: "sheetIn 260ms cubic-bezier(0.2, 0.9, 0.2, 1) forwards",
        }}
      >
        <div
          style={{
            padding: "20px 24px 16px",
            borderBottom: "1px solid var(--vf-panel-stroke)",
            display: "flex",
            alignItems: "flex-start",
            justifyContent: "space-between",
            flexShrink: 0,
          }}
        >
          <div>
            <div
              style={{
                fontSize: 10,
                letterSpacing: "0.22em",
                color: "var(--vf-text-muted)",
                fontFamily: "var(--font-mono)",
                textTransform: "uppercase",
                marginBottom: 6,
              }}
            >
              {t.queueOverlay.kicker}
            </div>
            <div
              style={{
                fontSize: 22,
                fontWeight: 600,
                letterSpacing: "-0.01em",
                color: "var(--vf-text)",
              }}
            >
              {t.queueOverlay.title}
            </div>
            <div
              style={{
                marginTop: 6,
                fontFamily: "var(--font-mono)",
                fontSize: 11,
                color: "var(--vf-text-dim)",
                lineHeight: 1.6,
              }}
            >
              {t.queueOverlay.description}
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            style={{
              width: 36,
              height: 36,
              borderRadius: "50%",
              border: "1px solid var(--vf-panel-stroke)",
              background: "rgb(var(--lt-ink, 255 255 255) / 0.03)",
              color: "var(--vf-text)",
              fontSize: 20,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              cursor: "pointer",
              flexShrink: 0,
            }}
          >
            ×
          </button>
        </div>

        <div
          style={{
            flex: 1,
            overflowY: "auto",
            padding: "16px 24px 24px",
            display: "flex",
            flexDirection: "column",
            gap: 10,
          }}
        >
          {error && (
            <div
              style={{
                padding: "10px 12px",
                border: "1px solid oklch(0.74 0.18 22 / 0.4)",
                background: "oklch(0.74 0.18 22 / 0.08)",
                borderRadius: 10,
                fontFamily: "var(--font-mono)",
                fontSize: 11,
                color: "oklch(calc(0.82 - var(--lt-flip, 0) * 0.34) 0.14 22)",
              }}
            >
              {error}
            </div>
          )}

          {jobs.length === 0 && !showLanes && (
            <div
              style={{
                padding: "28px 12px",
                textAlign: "center",
                fontFamily: "var(--font-mono)",
                fontSize: 12,
                color: "var(--vf-text-muted)",
                lineHeight: 1.7,
              }}
            >
              {t.queueOverlay.emptyTitle}
              <br />
              {t.queueOverlay.emptyHint}
            </div>
          )}

          <QueueLaneList
            lanes={shown}
            showLanes={showLanes}
            showProfile={showProfile}
            busyId={busyId}
            onCancel={(runId) => void cancel(runId)}
            onWatch={onWatch}
          />
        </div>
      </div>
    </div>
  );
}

export function JobRow({
  job,
  showProfile = false,
  running = false,
  stoppable = true,
  position,
  onCancel,
  onWatch,
  cancelling = false,
}: {
  job: QueuedJobInfo;
  /** Say whose folders the job writes into (ADR-114); off on a one-person
   *  install, where every job is the default profile's. */
  showProfile?: boolean;
  running?: boolean;
  /** False for a running job the server cannot stop part-way. */
  stoppable?: boolean;
  position?: number;
  onCancel: () => void;
  /** Only the running job has a log to watch; one still waiting has none yet. */
  onWatch?: () => void;
  cancelling?: boolean;
}) {
  const t = useT();
  return (
    <div
      style={{
        display: "flex",
        flexWrap: "wrap",
        alignItems: "center",
        gap: 14,
        padding: "12px 14px",
        borderRadius: 12,
        border: running
          ? "1px solid var(--accent-vf)"
          : "1px solid var(--vf-panel-stroke)",
        background: running
          ? "linear-gradient(180deg, var(--accent-soft) 0%, rgb(var(--lt-surface, 12 14 18) / 0.5) 100%)"
          : "rgb(var(--lt-ink, 255 255 255) / 0.025)",
      }}
    >
      <span
        style={{
          width: 34,
          textAlign: "center",
          fontFamily: "var(--font-mono)",
          fontSize: running ? 14 : 12,
          color: running ? "var(--accent-vf)" : "var(--vf-text-muted)",
          flexShrink: 0,
        }}
      >
        {running ? "▶" : t.queueOverlay.position(position ?? 0)}
      </span>

      <div style={{ flex: "1 1 180px", minWidth: 0 }}>
        <div
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: 13,
            color: "var(--vf-text)",
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}
        >
          {job.label}
        </div>
        <div
          style={{
            marginTop: 3,
            fontFamily: "var(--font-mono)",
            fontSize: 10,
            letterSpacing: "0.06em",
            textTransform: "uppercase",
            color: "var(--vf-text-muted)",
          }}
        >
          {jobOrigin(t, job)} · {taskLabel(t, job.task)} · {strategyLabel(t, job.strategy)} ·{" "}
          {showProfile && (
            <>{t.profile.queueOwner(jobProfileLabel(job, t.profile.defaultShort))} · </>
          )}
          {running
            ? t.queueOverlay.running
            : t.queueOverlay.waiting(waitedFor(job.submitted_at))}
        </div>
        {/* In the row, not only in the button's tooltip: a disabled button is
            out of reach of the keyboard and of screen readers. */}
        {running && !stoppable && (
          <div
            style={{
              marginTop: 5,
              fontFamily: "var(--font-mono)",
              fontSize: 11,
              lineHeight: 1.5,
              color: "var(--vf-text-dim)",
            }}
          >
            {t.queueOverlay.stopUnavailable}
          </div>
        )}
      </div>

      {running && onWatch && (
        <button
          type="button"
          onClick={onWatch}
          style={{
            padding: "7px 12px",
            borderRadius: 9,
            border: "1px solid var(--accent-vf)",
            background: "var(--accent-soft)",
            color: "var(--vf-text)",
            fontFamily: "var(--font-mono)",
            fontSize: 11,
            cursor: "pointer",
            flexShrink: 0,
          }}
        >
          {t.queueOverlay.watch}
        </button>
      )}

      <button
        type="button"
        onClick={onCancel}
        disabled={cancelling || (running && !stoppable)}
        title={
          running
            ? stoppable
              ? t.queueOverlay.stopTitle
              : t.queueOverlay.stopUnavailable
            : t.queueOverlay.removeTitle
        }
        style={{
          padding: "7px 12px",
          borderRadius: 9,
          border: "1px solid var(--vf-panel-stroke)",
          background: "rgb(var(--lt-ink, 255 255 255) / 0.03)",
          color: "var(--vf-text-dim)",
          fontFamily: "var(--font-mono)",
          fontSize: 11,
          cursor: cancelling
            ? "wait"
            : running && !stoppable
              ? "not-allowed"
              : "pointer",
          opacity: cancelling || (running && !stoppable) ? 0.5 : 1,
          flexShrink: 0,
        }}
      >
        {cancelling ? "…" : running ? t.queueOverlay.stop : t.queueOverlay.remove}
      </button>
    </div>
  );
}

/** The heading of one queue: this machine, or the Colab of one account. */
function laneTitle(t: Dict, lane: string): string {
  const account = laneAccount(lane);
  return account === null ? t.queueOverlay.laneLocal : t.queueOverlay.laneColab(account);
}

/** Where a job runs, at the head of its row. */
function jobOrigin(t: Dict, job: QueuedJobInfo): string {
  return (job.lane ?? LOCAL_LANE) === LOCAL_LANE
    ? t.queueOverlay.originLocal
    : t.queueOverlay.originColab;
}

/** The queues, one section each, with every button bound to its own row's job:
 *  stopping the Colab job never touches the local one (ADR-163 phase 4). */
export function QueueLaneList({
  lanes,
  showLanes,
  showProfile,
  busyId,
  onCancel,
  onWatch,
}: {
  lanes: QueueLane[];
  /** Headings per queue; off when the local queue is the only one. */
  showLanes: boolean;
  showProfile: boolean;
  busyId: string | null;
  onCancel: (runId: string) => void;
  onWatch?: (job: QueuedJobInfo) => void;
}) {
  const t = useT();
  return (
    <>
      {lanes.map(({ lane, active, pending }) => (
        <section
          key={lane}
          aria-label={laneTitle(t, lane)}
          data-lane={lane}
          style={{ display: "flex", flexDirection: "column", gap: 10 }}
        >
          {showLanes && (
            <div
              style={{
                marginTop: 6,
                display: "flex",
                alignItems: "baseline",
                justifyContent: "space-between",
                gap: 12,
                fontFamily: "var(--font-mono)",
                fontSize: 10,
                letterSpacing: "0.16em",
                textTransform: "uppercase",
                color: "var(--vf-text-muted)",
              }}
            >
              <span>{laneTitle(t, lane)}</span>
              {active === null && pending.length === 0 && (
                <span style={{ letterSpacing: "0.06em" }}>{t.queueOverlay.laneEmpty}</span>
              )}
            </div>
          )}

          {/* The running job can be stopped too (ADR-088): it finishes the epoch
              it is in and keeps the best checkpoint, so the button is safe — for
              the kinds of run that read the stop at all (lib/run-control.ts). */}
          {active && (
            <JobRow
              job={active}
              showProfile={showProfile}
              running
              stoppable={stopMode(active) !== "none"}
              onWatch={onWatch ? () => onWatch(active) : undefined}
              onCancel={() => onCancel(active.run_id)}
              cancelling={busyId === active.run_id}
            />
          )}

          {pending.map((job, index) => (
            <JobRow
              key={job.run_id}
              job={job}
              showProfile={showProfile}
              position={index + 1}
              onCancel={() => onCancel(job.run_id)}
              cancelling={busyId === job.run_id}
            />
          ))}
        </section>
      ))}
    </>
  );
}
