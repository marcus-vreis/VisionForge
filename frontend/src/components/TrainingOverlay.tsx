import { useEffect, useRef, useState } from "react";
import type { RunStatus } from "../types/run";

interface TrainingOverlayProps {
  status: RunStatus;
  taskAccent: string;
  taskLabel: string;
  onClose: () => void;
  onViewResults?: () => void;
}

/** Modal overlay shown while an experiment is running or just completed. */
export function TrainingOverlay({
  status,
  taskAccent,
  taskLabel,
  onClose,
  onViewResults,
}: TrainingOverlayProps) {
  const isRunning = status.status === "running";
  const isCompleted = status.status === "completed";
  const hasFailed = status.status === "failed";
  const isFinished = isCompleted || hasFailed;

  const [fakeProgress, setFakeProgress] = useState(0);
  const [logs, setLogs] = useState<string[]>([
    `$ visionforge train --task classification`,
    `> inicializando runtime · ${status.run_id ?? "..."}`,
    `> carregando dataset…`,
  ]);
  const logRef = useRef<HTMLDivElement>(null);
  // Tracks which terminal status we already processed so the effect only runs once
  const handledStatusRef = useRef<string | null>(null);

  // Synthetic progress animation while running
  useEffect(() => {
    if (!isRunning) return;
    const id = setInterval(() => {
      setFakeProgress((p) => Math.min(p + 0.004, 0.92));
      const now = new Date().toLocaleTimeString("pt-BR", {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      });
      setLogs((prev) => [...prev.slice(-24), `> running · polled at ${now}`]);
    }, 2000);
    return () => clearInterval(id);
  }, [isRunning]);

  // Handle terminal states — subscribe to status.status changes as an external signal
  useEffect(() => {
    if (handledStatusRef.current === status.status) return;
    if (isCompleted) {
      handledStatusRef.current = status.status;
      const timer = setTimeout(() => {
        setFakeProgress(1);
        setLogs((prev) => [...prev.slice(-24), "$ training complete"]);
      }, 0);
      return () => clearTimeout(timer);
    }
    if (hasFailed) {
      handledStatusRef.current = status.status;
      const msg = status.error ?? "erro desconhecido";
      const timer = setTimeout(() => {
        setLogs((prev) => [
          ...prev.slice(-24),
          `$ training failed · ${msg}`,
        ]);
      }, 0);
      return () => clearTimeout(timer);
    }
  }, [status.status, isCompleted, hasFailed, status.error]);

  // Auto-scroll logs
  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logs]);

  const pct = isCompleted ? 100 : Math.round(fakeProgress * 100);

  return (
    <div
      onClick={(e) => {
        if (e.target === e.currentTarget && isFinished) onClose();
      }}
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 90,
        background: "rgba(4,5,7,0.78)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        animation: "overlayIn 220ms ease forwards",
        padding: 24,
      }}
    >
      <div
        style={{
          width: "min(720px, 100%)",
          background: "rgba(12,14,18,0.95)",
          border: "1px solid var(--vf-panel-stroke)",
          borderRadius: 18,
          padding: 28,
          animation: "sheetIn 260ms cubic-bezier(0.2, 0.9, 0.2, 1) forwards",
        }}
      >
        {/* Header */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 14,
            marginBottom: 18,
          }}
        >
          <span
            style={{
              width: 16,
              height: 16,
              borderRadius: "50%",
              border: `2px solid ${taskAccent}`,
              borderTopColor: isFinished ? taskAccent : "transparent",
              animation: isFinished ? "none" : "spin 0.8s linear infinite",
              flexShrink: 0,
            }}
          />
          <div style={{ flex: 1 }}>
            <div
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: 10,
                letterSpacing: "0.22em",
                textTransform: "uppercase",
                color: "var(--vf-text-muted)",
              }}
            >
              {isFinished
                ? hasFailed
                  ? "training failed"
                  : "training complete"
                : `training · ${taskLabel}`}
            </div>
            <div
              style={{
                fontSize: 18,
                fontWeight: 600,
                marginTop: 4,
                fontFamily: "var(--font-mono)",
                color: "var(--vf-text)",
              }}
            >
              {status.run_id ?? "iniciando…"}
            </div>
          </div>
          <div
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: 28,
              fontWeight: 600,
              color: hasFailed ? "oklch(0.74 0.18 22)" : taskAccent,
            }}
          >
            {pct}%
          </div>
        </div>

        {/* Progress bar */}
        <div
          style={{
            height: 6,
            borderRadius: 999,
            overflow: "hidden",
            background: "rgba(255,255,255,0.04)",
            marginBottom: 18,
          }}
        >
          <div
            style={{
              width: `${pct}%`,
              height: "100%",
              background: hasFailed
                ? `linear-gradient(90deg, oklch(0.74 0.18 22), oklch(0.74 0.18 22 / 0.6))`
                : `linear-gradient(90deg, ${taskAccent}, ${taskAccent}aa)`,
              boxShadow: `0 0 12px ${taskAccent}`,
              transition: "width 320ms ease",
            }}
          />
        </div>

        {/* Failure detail panel — shown above logs when failed */}
        {hasFailed && (
          <div
            style={{
              padding: "14px 16px",
              marginBottom: 14,
              background: "oklch(0.704 0.191 22.216 / 0.10)",
              border: "1px solid oklch(0.704 0.191 22.216 / 0.45)",
              borderRadius: 12,
              fontFamily: "var(--font-mono)",
              fontSize: 12.5,
              color: "oklch(0.88 0.14 22)",
              whiteSpace: "pre-wrap",
              wordBreak: "break-word",
              lineHeight: 1.55,
              maxHeight: 180,
              overflowY: "auto",
            }}
          >
            <div
              style={{
                fontSize: 10,
                letterSpacing: "0.18em",
                textTransform: "uppercase",
                color: "oklch(0.7 0.18 22)",
                marginBottom: 6,
              }}
            >
              Detalhe do erro
            </div>
            {status.error ?? "Erro desconhecido — verifique os logs do servidor."}
          </div>
        )}

        {/* Log stream */}
        <div
          ref={logRef}
          style={{
            height: hasFailed ? 120 : 220,
            padding: 16,
            background: "rgba(0,0,0,0.45)",
            border: "1px solid var(--vf-panel-stroke)",
            borderRadius: 12,
            fontFamily: "var(--font-mono)",
            fontSize: 12,
            color: "var(--vf-text-dim)",
            overflowY: "auto",
            lineHeight: 1.6,
          }}
        >
          {logs.map((line, i) => (
            <div
              key={i}
              style={{
                color: line.startsWith("$")
                  ? taskAccent
                  : line.startsWith(">")
                    ? "var(--vf-text)"
                    : "var(--vf-text-muted)",
              }}
            >
              {line}
            </div>
          ))}
        </div>

        {/* Action buttons */}
        <div
          style={{
            display: "flex",
            gap: 10,
            marginTop: 18,
            justifyContent: "flex-end",
          }}
        >
          {isRunning && (
            <button
              type="button"
              onClick={onClose}
              style={{
                padding: "10px 18px",
                background: "transparent",
                border: "1px solid var(--vf-panel-stroke)",
                borderRadius: 10,
                color: "var(--vf-text-dim)",
                fontFamily: "var(--font-mono)",
                fontSize: 11,
                letterSpacing: "0.08em",
                textTransform: "uppercase",
                cursor: "pointer",
              }}
            >
              Minimizar
            </button>
          )}
          {isFinished && (
            <>
              <button
                type="button"
                onClick={onClose}
                style={{
                  padding: "10px 18px",
                  background: "transparent",
                  border: "1px solid var(--vf-panel-stroke)",
                  borderRadius: 10,
                  color: "var(--vf-text-dim)",
                  fontFamily: "var(--font-mono)",
                  fontSize: 11,
                  letterSpacing: "0.08em",
                  textTransform: "uppercase",
                  cursor: "pointer",
                }}
              >
                Fechar
              </button>
              {!hasFailed && onViewResults && (
                <button
                  type="button"
                  onClick={onViewResults}
                  style={{
                    padding: "10px 18px",
                    background: "var(--accent-soft)",
                    border: `1px solid ${taskAccent}`,
                    borderRadius: 10,
                    color: "var(--vf-text)",
                    fontFamily: "var(--font-mono)",
                    fontSize: 11,
                    letterSpacing: "0.08em",
                    textTransform: "uppercase",
                    boxShadow: "inset 0 0 14px var(--accent-glow)",
                    cursor: "pointer",
                  }}
                >
                  ↗ Ver resultados
                </button>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
