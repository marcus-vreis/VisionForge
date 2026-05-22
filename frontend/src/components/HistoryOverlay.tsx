import { useEffect, useState } from "react";
import { fetchRuns } from "../api/client";
import type { RunSummary } from "../types/run";
import { RunDetailPanel } from "./RunDetailPanel";

interface HistoryOverlayProps {
  open: boolean;
  onClose: () => void;
  onCountChange?: (count: number) => void;
}

/** Color accent per task type, mirroring the VisionForge oklch palette. */
const TASK_ACCENT: Record<string, string> = {
  classification: "oklch(0.74 0.18 22)",
  detection: "oklch(0.78 0.18 150)",
  regression: "oklch(0.74 0.16 240)",
  segmentation: "oklch(0.74 0.18 305)",
};

/** Status dot color — completed is muted green, running is accent, failed is red. */
function statusColor(status: string): string {
  if (status === "completed") return "oklch(0.78 0.18 150)";
  if (status === "running") return "oklch(0.74 0.16 240)";
  if (status === "failed") return "oklch(0.74 0.18 22)";
  return "var(--vf-text-muted)";
}

/** Format an ISO date string using pt-BR locale. */
function fmtDate(iso: string): string {
  try {
    return new Date(iso).toLocaleString("pt-BR", {
      day: "2-digit",
      month: "2-digit",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return iso;
  }
}

/** Metric keys shown on each card when present. */
const METRIC_KEYS = ["accuracy", "f1", "val_loss"];

/** One run card inside the history list. */
function RunCard({ run, onClick }: { run: RunSummary; onClick: () => void }) {
  const accent = TASK_ACCENT[run.task] ?? "var(--vf-text-muted)";
  const dot = statusColor(run.status);
  const shownMetrics = METRIC_KEYS.filter(
    (k) => run.final_metrics[k] !== undefined,
  );

  return (
    <button
      type="button"
      onClick={onClick}
      style={{
        padding: "14px 18px",
        background: "rgba(255,255,255,0.025)",
        border: "1px solid var(--vf-panel-stroke)",
        borderRadius: 12,
        display: "flex",
        flexDirection: "column",
        gap: 8,
        cursor: "pointer",
        textAlign: "left",
        width: "100%",
        color: "inherit",
      }}
    >
      {/* Top row: name + status dot */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 8,
        }}
      >
        <div
          style={{
            fontWeight: 600,
            fontSize: 14,
            color: "var(--vf-text)",
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}
        >
          {run.experiment_name}
        </div>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 6,
            flexShrink: 0,
          }}
        >
          <span
            style={{
              width: 7,
              height: 7,
              borderRadius: "50%",
              background: dot,
              boxShadow: `0 0 8px ${dot}`,
              flexShrink: 0,
            }}
          />
          <span
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: 10,
              letterSpacing: "0.12em",
              textTransform: "uppercase",
              color: dot,
            }}
          >
            {run.status}
          </span>
        </div>
      </div>

      {/* Second row: arch + task pill + epochs */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          flexWrap: "wrap",
        }}
      >
        <span
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: 11,
            color: "var(--vf-text-dim)",
          }}
        >
          {run.model_arch}
        </span>
        <span
          style={{
            padding: "2px 8px",
            background: `${accent}1a`,
            border: `1px solid ${accent}55`,
            borderRadius: 999,
            fontFamily: "var(--font-mono)",
            fontSize: 10,
            color: accent,
            letterSpacing: "0.10em",
            textTransform: "uppercase",
          }}
        >
          {run.task}
        </span>
        <span
          style={{
            marginLeft: "auto",
            fontFamily: "var(--font-mono)",
            fontSize: 11,
            color: "var(--vf-text-muted)",
          }}
        >
          {run.epochs_completed} epoch{run.epochs_completed !== 1 ? "s" : ""}
        </span>
      </div>

      {/* Metrics row */}
      {shownMetrics.length > 0 && (
        <div
          style={{
            display: "flex",
            gap: 14,
            paddingTop: 4,
            borderTop: "1px solid var(--vf-panel-stroke)",
          }}
        >
          {shownMetrics.map((k) => (
            <div key={k} style={{ display: "flex", flexDirection: "column", gap: 1 }}>
              <span
                style={{
                  fontFamily: "var(--font-mono)",
                  fontSize: 9,
                  letterSpacing: "0.16em",
                  textTransform: "uppercase",
                  color: "var(--vf-text-muted)",
                }}
              >
                {k}
              </span>
              <span
                style={{
                  fontFamily: "var(--font-mono)",
                  fontSize: 13,
                  fontWeight: 600,
                  color: accent,
                }}
              >
                {run.final_metrics[k].toFixed(4)}
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Date row */}
      <div
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: 10,
          color: "var(--vf-text-muted)",
          letterSpacing: "0.06em",
        }}
      >
        {fmtDate(run.started_at)}
        {run.finished_at ? ` → ${fmtDate(run.finished_at)}` : " · em andamento"}
      </div>
    </button>
  );
}

/** History overlay — fetches and displays past experiment runs from /api/runs. */
export function HistoryOverlay({ open, onClose, onCountChange }: HistoryOverlayProps) {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    setLoading(true);
    setError(null);
    setSelectedRunId(null);
    fetchRuns()
      .then((data) => {
        setRuns(data);
        onCountChange?.(data.length);
      })
      .catch((e: unknown) => {
        const msg =
          e instanceof Error ? e.message : "Erro ao carregar histórico.";
        setError(msg);
      })
      .finally(() => setLoading(false));
  }, [open]); // eslint-disable-line react-hooks/exhaustive-deps

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
          width: selectedRunId ? "min(960px, 100%)" : "min(640px, 100%)",
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
        {/* Header */}
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
              // training history
            </div>
            <div
              style={{
                fontSize: 22,
                fontWeight: 600,
                letterSpacing: "-0.01em",
                color: "var(--vf-text)",
              }}
            >
              Treinamentos recentes
              {runs.length > 0 && (
                <span
                  style={{
                    marginLeft: 10,
                    padding: "2px 9px",
                    background: "rgba(255,255,255,0.06)",
                    border: "1px solid var(--vf-panel-stroke)",
                    borderRadius: 999,
                    fontSize: 13,
                    fontWeight: 500,
                    color: "var(--vf-text-dim)",
                    verticalAlign: "middle",
                  }}
                >
                  {runs.length}
                </span>
              )}
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

        {/* Body */}
        <div style={{ flex: 1, overflowY: "auto", padding: "16px 24px 24px" }}>
          {/* Loading state */}
          {loading && (
            <div
              style={{
                padding: 48,
                textAlign: "center",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                gap: 16,
              }}
            >
              <span
                style={{
                  width: 22,
                  height: 22,
                  borderRadius: "50%",
                  border: "2px solid var(--vf-panel-stroke)",
                  borderTopColor: "var(--vf-text-dim)",
                  animation: "spin 0.8s linear infinite",
                  display: "block",
                }}
              />
              <span
                style={{
                  fontFamily: "var(--font-mono)",
                  fontSize: 12,
                  color: "var(--vf-text-muted)",
                  letterSpacing: "0.08em",
                }}
              >
                carregando histórico…
              </span>
            </div>
          )}

          {/* Error state */}
          {!loading && error !== null && (
            <div
              style={{
                padding: "14px 18px",
                background: "oklch(0.704 0.191 22.216 / 0.10)",
                border: "1px solid oklch(0.704 0.191 22.216 / 0.4)",
                borderRadius: 12,
                fontFamily: "var(--font-mono)",
                fontSize: 13,
                color: "oklch(0.85 0.14 22)",
                lineHeight: 1.55,
              }}
            >
              <div
                style={{
                  fontSize: 10,
                  letterSpacing: "0.16em",
                  textTransform: "uppercase",
                  color: "oklch(0.7 0.18 22)",
                  marginBottom: 4,
                }}
              >
                Erro
              </div>
              {error}
            </div>
          )}

          {/* Empty state */}
          {!loading && error === null && runs.length === 0 && (
            <div
              style={{
                padding: 48,
                textAlign: "center",
              }}
            >
              <div
                style={{
                  fontSize: 40,
                  color: "var(--vf-text-muted)",
                  marginBottom: 16,
                }}
              >
                ◇
              </div>
              <div
                style={{
                  fontFamily: "var(--font-mono)",
                  fontSize: 14,
                  color: "var(--vf-text-dim)",
                  marginBottom: 8,
                }}
              >
                Nenhum treinamento ainda
              </div>
              <div
                style={{
                  fontFamily: "var(--font-mono)",
                  fontSize: 12,
                  color: "var(--vf-text-muted)",
                }}
              >
                Execute o primeiro experimento para vê-lo aqui.
              </div>
            </div>
          )}

          {/* Detail panel takes over when a run is selected. */}
          {selectedRunId && (
            <RunDetailPanel
              runId={selectedRunId}
              onBack={() => setSelectedRunId(null)}
            />
          )}

          {/* Run list */}
          {!selectedRunId && !loading && error === null && runs.length > 0 && (
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              {runs.map((run) => (
                <RunCard
                  key={run.run_id}
                  run={run}
                  onClick={() => setSelectedRunId(run.run_id)}
                />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
