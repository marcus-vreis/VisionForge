import { useEffect, useState } from "react";
import { fetchRuns } from "../api/client";
import type { RunSummary } from "../types/run";
import { CompareRunsPanel } from "./CompareRunsPanel";
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
function RunCard({
  run,
  onClick,
  selectable,
  selected,
  onToggleSelect,
}: {
  run: RunSummary;
  onClick: () => void;
  selectable?: boolean;
  selected?: boolean;
  onToggleSelect?: () => void;
}) {
  const accent = TASK_ACCENT[run.task] ?? "var(--vf-text-muted)";
  const dot = statusColor(run.status);
  const shownMetrics = METRIC_KEYS.filter(
    (k) => run.final_metrics[k] !== undefined,
  );

  const handleClick = () => {
    if (selectable && onToggleSelect) {
      onToggleSelect();
    } else {
      onClick();
    }
  };

  return (
    <button
      type="button"
      onClick={handleClick}
      style={{
        position: "relative",
        padding: "14px 18px",
        background: selected
          ? "rgba(120, 200, 130, 0.10)"
          : "rgba(255,255,255,0.025)",
        border: `1px solid ${selected ? "oklch(0.78 0.18 150)" : "var(--vf-panel-stroke)"}`,
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
      {selectable && (
        <span
          style={{
            position: "absolute",
            top: 8,
            right: 8,
            width: 18,
            height: 18,
            borderRadius: 4,
            border: `1.5px solid ${selected ? "oklch(0.78 0.18 150)" : "var(--vf-panel-stroke)"}`,
            background: selected ? "oklch(0.78 0.18 150 / 0.3)" : "transparent",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: 11,
            color: "var(--vf-text)",
          }}
        >
          {selected ? "✓" : ""}
        </span>
      )}
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
        {run.preprocessing_count !== undefined && run.preprocessing_count > 0 && (
          <span
            title={`${run.preprocessing_count} filtro(s) de pré-processamento aplicados ao treino`}
            style={{
              padding: "2px 8px",
              background: "oklch(0.72 0.16 150 / 0.14)",
              border: "1px solid oklch(0.72 0.16 150 / 0.45)",
              borderRadius: 999,
              fontFamily: "var(--font-mono)",
              fontSize: 10,
              color: "oklch(0.88 0.15 150)",
              letterSpacing: "0.10em",
              textTransform: "uppercase",
            }}
          >
            ⚗ {run.preprocessing_count} filtro
            {run.preprocessing_count === 1 ? "" : "s"}
          </span>
        )}
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
  const [compareMode, setCompareMode] = useState(false);
  const [compareSelection, setCompareSelection] = useState<string[]>([]);
  const [compareActiveIds, setCompareActiveIds] = useState<string[] | null>(null);

  useEffect(() => {
    if (!open) return;
    setLoading(true);
    setError(null);
    setSelectedRunId(null);
    setCompareMode(false);
    setCompareSelection([]);
    setCompareActiveIds(null);
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

  const toggleSelect = (runId: string) => {
    setCompareSelection((prev) =>
      prev.includes(runId) ? prev.filter((id) => id !== runId) : [...prev, runId],
    );
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
          width:
            selectedRunId || compareActiveIds
              ? "min(960px, 100%)"
              : "min(640px, 100%)",
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
          <div style={{ display: "flex", alignItems: "center", gap: 10, flexShrink: 0 }}>
            {!selectedRunId && !compareActiveIds && runs.length >= 2 && (
              <button
                type="button"
                onClick={() => {
                  setCompareMode((m) => !m);
                  if (compareMode) setCompareSelection([]);
                }}
                style={{
                  padding: "8px 14px",
                  background: compareMode ? "oklch(0.78 0.18 150 / 0.18)" : "rgba(255,255,255,0.04)",
                  border: `1px solid ${compareMode ? "oklch(0.78 0.18 150)" : "var(--vf-panel-stroke)"}`,
                  borderRadius: 10,
                  color: compareMode ? "oklch(0.88 0.16 150)" : "var(--vf-text-dim)",
                  fontFamily: "var(--font-mono)",
                  fontSize: 11,
                  letterSpacing: "0.10em",
                  textTransform: "uppercase",
                  cursor: "pointer",
                }}
              >
                {compareMode ? "Cancelar comparação" : "↔ Comparar"}
              </button>
            )}
            {compareMode && compareSelection.length >= 2 && (
              <button
                type="button"
                onClick={() => setCompareActiveIds(compareSelection)}
                style={{
                  padding: "8px 14px",
                  background: "oklch(0.78 0.18 150 / 0.30)",
                  border: "1px solid oklch(0.78 0.18 150)",
                  borderRadius: 10,
                  color: "var(--vf-text)",
                  fontFamily: "var(--font-mono)",
                  fontSize: 11,
                  letterSpacing: "0.10em",
                  textTransform: "uppercase",
                  cursor: "pointer",
                  fontWeight: 600,
                }}
              >
                Comparar {compareSelection.length}
              </button>
            )}
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
              }}
            >
              ×
            </button>
          </div>
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

          {/* Compare panel takes over when 2+ runs selected and comparison was confirmed. */}
          {compareActiveIds && !selectedRunId && (
            <CompareRunsPanel
              runIds={compareActiveIds}
              onBack={() => {
                setCompareActiveIds(null);
                setCompareMode(false);
                setCompareSelection([]);
              }}
            />
          )}

          {/* Detail panel takes over when a run is selected. */}
          {selectedRunId && !compareActiveIds && (
            <RunDetailPanel
              runId={selectedRunId}
              onBack={() => setSelectedRunId(null)}
            />
          )}

          {/* Compare mode tip */}
          {!selectedRunId && !compareActiveIds && compareMode && (
            <div
              style={{
                padding: "10px 14px",
                marginBottom: 12,
                background: "oklch(0.78 0.18 150 / 0.08)",
                border: "1px dashed oklch(0.78 0.18 150 / 0.5)",
                borderRadius: 10,
                fontFamily: "var(--font-mono)",
                fontSize: 11,
                color: "oklch(0.88 0.16 150)",
              }}
            >
              Modo comparação ativo — marque 2 ou mais runs e clique em
              "Comparar N".
            </div>
          )}

          {/* Run list */}
          {!selectedRunId && !compareActiveIds && !loading && error === null && runs.length > 0 && (
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              {runs.map((run) => (
                <RunCard
                  key={run.run_id}
                  run={run}
                  onClick={() => setSelectedRunId(run.run_id)}
                  selectable={compareMode}
                  selected={compareSelection.includes(run.run_id)}
                  onToggleSelect={() => toggleSelect(run.run_id)}
                />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
