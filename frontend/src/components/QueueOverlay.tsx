import { useCallback, useEffect, useState } from "react";

import { ApiError, cancelQueuedRun, fetchQueue } from "../api/client";
import { strategyLabel, taskLabel, waitedFor } from "../lib/queue-format";
import type { QueuedJobInfo } from "../types/run";

interface QueueOverlayProps {
  open: boolean;
  onClose: () => void;
  /** Reports the pending count up so the bottom-bar badge cannot go stale —
   *  cancelling here changes a number this component does not own. */
  onCountChange?: (pending: number) => void;
}

/** The run queue as a surface of its own (ADR-075).
 *
 * The training overlay shows the job you just submitted; this shows the whole
 * line, which is what you want after queueing an evening's work and coming back
 * to it. Only pending jobs can be dropped — a running one has no cooperative
 * stop point, so the backend answers 404 and the button is not offered.
 */
export function QueueOverlay({
  open,
  onClose,
  onCountChange,
}: QueueOverlayProps) {
  const [active, setActive] = useState<QueuedJobInfo | null>(null);
  const [pending, setPending] = useState<QueuedJobInfo[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const snap = await fetchQueue();
      setActive(snap.active);
      setPending(snap.pending);
      onCountChange?.(snap.pending.length);
      setError(null);
    } catch (e) {
      setError(
        e instanceof ApiError ? e.message : "Não foi possível ler a fila.",
      );
    }
  }, [onCountChange]);

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
          ? e.message
          : "Não foi possível cancelar esse treino.",
      );
    } finally {
      setBusyId(null);
    }
  };

  if (!open) return null;

  return (
    <div
      onClick={onClose}
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 100,
        background: "rgba(4,5,7,0.72)",
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
          background: "rgba(12,14,18,0.95)",
          border: "1px solid var(--vf-panel-stroke)",
          borderRadius: 18,
          boxShadow: "0 50px 120px rgba(0,0,0,0.7)",
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
              // fila
            </div>
            <div
              style={{
                fontSize: 22,
                fontWeight: 600,
                letterSpacing: "-0.01em",
                color: "var(--vf-text)",
              }}
            >
              Treinos na fila
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
              Uma GPU, um treino por vez. Submeta quantos quiser — eles rodam em
              ordem de envio, sozinhos.
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
              background: "rgba(255,255,255,0.03)",
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
                color: "oklch(0.82 0.14 22)",
              }}
            >
              {error}
            </div>
          )}

          {active === null && pending.length === 0 && (
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
              A GPU está livre e nada está esperando.
              <br />
              Envie um treino e, se enviar outro em seguida, ele aparece aqui.
            </div>
          )}

          {active && <JobRow job={active} running />}

          {pending.map((job, index) => (
            <JobRow
              key={job.run_id}
              job={job}
              position={index + 1}
              onCancel={() => void cancel(job.run_id)}
              cancelling={busyId === job.run_id}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

function JobRow({
  job,
  running = false,
  position,
  onCancel,
  cancelling = false,
}: {
  job: QueuedJobInfo;
  running?: boolean;
  position?: number;
  onCancel?: () => void;
  cancelling?: boolean;
}) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 14,
        padding: "12px 14px",
        borderRadius: 12,
        border: running
          ? "1px solid var(--accent-vf)"
          : "1px solid var(--vf-panel-stroke)",
        background: running
          ? "linear-gradient(180deg, var(--accent-soft) 0%, rgba(12,14,18,0.5) 100%)"
          : "rgba(255,255,255,0.025)",
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
        {running ? "▶" : `${position}º`}
      </span>

      <div style={{ flex: 1, minWidth: 0 }}>
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
          {taskLabel(job.task)} · {strategyLabel(job.strategy)} ·{" "}
          {running ? "em execução" : `esperando ${waitedFor(job.submitted_at)}`}
        </div>
      </div>

      {onCancel ? (
        <button
          type="button"
          onClick={onCancel}
          disabled={cancelling}
          title="Remover da fila (não afeta treinos já iniciados)"
          style={{
            padding: "7px 12px",
            borderRadius: 9,
            border: "1px solid var(--vf-panel-stroke)",
            background: "rgba(255,255,255,0.03)",
            color: "var(--vf-text-dim)",
            fontFamily: "var(--font-mono)",
            fontSize: 11,
            cursor: cancelling ? "wait" : "pointer",
            opacity: cancelling ? 0.5 : 1,
            flexShrink: 0,
          }}
        >
          {cancelling ? "…" : "🗑 remover"}
        </button>
      ) : (
        <span
          title="Um treino em andamento não pode ser cancelado: os trainers não têm ponto de parada, e interromper deixaria a pasta do run pela metade."
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: 10,
            color: "var(--vf-text-muted)",
            cursor: "help",
            flexShrink: 0,
          }}
        >
          sem cancelar
        </span>
      )}
    </div>
  );
}
