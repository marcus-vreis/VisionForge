import { useEffect, useState } from "react";
import { deleteRun, fetchRuns } from "../api/client";
import type { RunSummary } from "../types/run";
import { CompareRunsPanel } from "./CompareRunsPanel";
import { RunDetailPanel } from "./RunDetailPanel";

interface HistoryOverlayProps {
  open: boolean;
  onClose: () => void;
  onCountChange?: (count: number) => void;
}

/** Small filter chip — used for the task-type filter row. */
function FilterChip({
  label,
  active,
  onClick,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      style={{
        padding: "6px 12px",
        background: active ? "var(--accent-soft)" : "rgba(255,255,255,0.04)",
        border: `1px solid ${active ? "var(--accent-vf)" : "var(--vf-panel-stroke)"}`,
        borderRadius: 999,
        fontFamily: "var(--font-mono)",
        fontSize: 10,
        letterSpacing: "0.12em",
        textTransform: "uppercase",
        color: active ? "var(--vf-text)" : "var(--vf-text-dim)",
        cursor: "pointer",
      }}
    >
      {label}
    </button>
  );
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
  onDelete,
  deleting,
}: {
  run: RunSummary;
  onClick: () => void;
  selectable?: boolean;
  selected?: boolean;
  onToggleSelect?: () => void;
  onDelete?: () => void;
  deleting?: boolean;
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
      {!selectable && onDelete && (
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            if (deleting) return;
            onDelete();
          }}
          disabled={deleting}
          title="Excluir este run permanentemente"
          style={{
            position: "absolute",
            top: 8,
            right: 8,
            width: 26,
            height: 26,
            borderRadius: 8,
            background: deleting ? "rgba(255,255,255,0.04)" : "transparent",
            border: "1px solid var(--vf-panel-stroke)",
            color: deleting ? "var(--vf-text-muted)" : "oklch(0.78 0.16 22)",
            fontSize: 12,
            cursor: deleting ? "wait" : "pointer",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            opacity: deleting ? 0.5 : 1,
          }}
        >
          {deleting ? "…" : "🗑"}
        </button>
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
        {run.block && run.block !== "classification" && (
          <span
            title={`Bloco de experimento: ${run.block}`}
            style={{
              padding: "2px 8px",
              background: "rgba(180, 140, 255, 0.12)",
              border: "1px solid rgba(180, 140, 255, 0.45)",
              borderRadius: 999,
              fontFamily: "var(--font-mono)",
              fontSize: 10,
              color: "rgba(220, 190, 255, 0.95)",
              letterSpacing: "0.10em",
              textTransform: "uppercase",
            }}
          >
            ⛓ {run.block}
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
  const [query, setQuery] = useState("");
  const [taskFilter, setTaskFilter] = useState<string>("all");
  const [pendingDelete, setPendingDelete] = useState<RunSummary | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    setLoading(true);
    setError(null);
    setSelectedRunId(null);
    setCompareMode(false);
    setCompareSelection([]);
    setCompareActiveIds(null);
    setQuery("");
    setTaskFilter("all");
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

  const confirmDelete = async () => {
    if (!pendingDelete) return;
    const id = pendingDelete.run_id;
    setDeletingId(id);
    setDeleteError(null);
    try {
      await deleteRun(id);
      // Remove from local list and adjust selections that referenced it.
      const next = runs.filter((r) => r.run_id !== id);
      setRuns(next);
      onCountChange?.(next.length);
      if (selectedRunId === id) setSelectedRunId(null);
      setCompareSelection((prev) => prev.filter((rid) => rid !== id));
      setPendingDelete(null);
    } catch (e: unknown) {
      const msg =
        e instanceof Error ? e.message : "Falha ao excluir o run.";
      setDeleteError(msg);
    } finally {
      setDeletingId(null);
    }
  };

  // Client-side filter — keeps the list responsive even with hundreds of runs.
  // Search is case-insensitive and matches against experiment name OR arch.
  const filteredRuns = (() => {
    if (runs.length === 0) return runs;
    const q = query.trim().toLowerCase();
    return runs.filter((r) => {
      if (taskFilter !== "all" && r.task !== taskFilter) return false;
      if (q === "") return true;
      return (
        r.experiment_name.toLowerCase().includes(q) ||
        r.model_arch.toLowerCase().includes(q) ||
        r.run_id.toLowerCase().includes(q)
      );
    });
  })();

  const availableTasks = Array.from(new Set(runs.map((r) => r.task))).sort();

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

          {/* Search + task filter row (only when there's a list to filter) */}
          {!selectedRunId && !compareActiveIds && !loading && error === null && runs.length > 0 && (
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 10,
                marginBottom: 12,
                flexWrap: "wrap",
              }}
            >
              <div style={{ position: "relative", flex: 1, minWidth: 200 }}>
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="🔍 buscar por nome, arquitetura ou run_id…"
                  style={{
                    width: "100%",
                    padding: "9px 36px 9px 12px",
                    background: "rgba(0,0,0,0.35)",
                    border: "1px solid var(--vf-panel-stroke)",
                    borderRadius: 10,
                    color: "var(--vf-text)",
                    fontFamily: "var(--font-mono)",
                    fontSize: 12,
                  }}
                />
                {query && (
                  <button
                    type="button"
                    onClick={() => setQuery("")}
                    title="Limpar busca"
                    style={{
                      position: "absolute",
                      right: 6,
                      top: "50%",
                      transform: "translateY(-50%)",
                      width: 24,
                      height: 24,
                      borderRadius: "50%",
                      background: "rgba(255,255,255,0.06)",
                      border: "1px solid var(--vf-panel-stroke)",
                      color: "var(--vf-text-dim)",
                      fontSize: 12,
                      cursor: "pointer",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                    }}
                  >
                    ×
                  </button>
                )}
              </div>
              {availableTasks.length > 1 && (
                <div style={{ display: "flex", gap: 6 }}>
                  <FilterChip
                    label="todos"
                    active={taskFilter === "all"}
                    onClick={() => setTaskFilter("all")}
                  />
                  {availableTasks.map((t) => (
                    <FilterChip
                      key={t}
                      label={t}
                      active={taskFilter === t}
                      onClick={() => setTaskFilter(t)}
                    />
                  ))}
                </div>
              )}
              {(query !== "" || taskFilter !== "all") && (
                <span
                  style={{
                    fontFamily: "var(--font-mono)",
                    fontSize: 10,
                    color: "var(--vf-text-muted)",
                    letterSpacing: "0.08em",
                  }}
                >
                  {filteredRuns.length} / {runs.length}
                </span>
              )}
            </div>
          )}

          {/* Run list */}
          {!selectedRunId && !compareActiveIds && !loading && error === null && runs.length > 0 && (
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              {filteredRuns.length === 0 ? (
                <div
                  style={{
                    padding: 24,
                    fontFamily: "var(--font-mono)",
                    fontSize: 12,
                    color: "var(--vf-text-muted)",
                    textAlign: "center",
                    border: "1px dashed var(--vf-panel-stroke)",
                    borderRadius: 10,
                  }}
                >
                  Nenhum run combina com o filtro atual.
                </div>
              ) : (
                filteredRuns.map((run) => (
                  <RunCard
                    key={run.run_id}
                    run={run}
                    onClick={() => setSelectedRunId(run.run_id)}
                    selectable={compareMode}
                    selected={compareSelection.includes(run.run_id)}
                    onToggleSelect={() => toggleSelect(run.run_id)}
                    onDelete={() => {
                      setDeleteError(null);
                      setPendingDelete(run);
                    }}
                    deleting={deletingId === run.run_id}
                  />
                ))
              )}
            </div>
          )}
        </div>
      </div>

      {/* Delete confirmation modal — separate layer above the history sheet
          so its backdrop click closes the dialog, not the history. */}
      {pendingDelete && (
        <div
          onClick={(e) => {
            if (e.target === e.currentTarget && !deletingId) {
              setPendingDelete(null);
              setDeleteError(null);
            }
          }}
          style={{
            position: "fixed",
            inset: 0,
            zIndex: 200,
            background: "rgba(2,3,5,0.78)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            padding: 24,
          }}
        >
          <div
            style={{
              width: "min(440px, 100%)",
              background: "rgba(12,14,18,0.98)",
              border: "1px solid var(--vf-panel-stroke)",
              borderRadius: 16,
              padding: 22,
              display: "flex",
              flexDirection: "column",
              gap: 14,
            }}
          >
            <div
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: 10,
                letterSpacing: "0.20em",
                textTransform: "uppercase",
                color: "oklch(0.78 0.16 22)",
              }}
            >
              // excluir run permanentemente
            </div>
            <div style={{ fontFamily: "var(--font-mono)", fontSize: 13, color: "var(--vf-text)" }}>
              {pendingDelete.experiment_name}
              <span style={{ color: "var(--vf-text-muted)", marginLeft: 6 }}>
                · {pendingDelete.model_arch}
              </span>
            </div>
            <div
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: 11,
                color: "var(--vf-text-dim)",
                lineHeight: 1.6,
              }}
            >
              A pasta do run, checkpoint e todos os plots/relatórios serão
              removidos do disco. Esta ação é irreversível.
            </div>
            {deleteError && (
              <div
                style={{
                  padding: "8px 12px",
                  background: "oklch(0.704 0.191 22.216 / 0.10)",
                  border: "1px solid oklch(0.704 0.191 22.216 / 0.4)",
                  borderRadius: 8,
                  fontFamily: "var(--font-mono)",
                  fontSize: 11,
                  color: "oklch(0.85 0.14 22)",
                }}
              >
                {deleteError}
              </div>
            )}
            <div style={{ display: "flex", gap: 10, justifyContent: "flex-end", marginTop: 4 }}>
              <button
                type="button"
                onClick={() => {
                  setPendingDelete(null);
                  setDeleteError(null);
                }}
                disabled={!!deletingId}
                style={{
                  padding: "9px 16px",
                  background: "transparent",
                  border: "1px solid var(--vf-panel-stroke)",
                  borderRadius: 10,
                  color: "var(--vf-text-dim)",
                  fontFamily: "var(--font-mono)",
                  fontSize: 11,
                  letterSpacing: "0.10em",
                  textTransform: "uppercase",
                  cursor: deletingId ? "not-allowed" : "pointer",
                  opacity: deletingId ? 0.5 : 1,
                }}
              >
                Cancelar
              </button>
              <button
                type="button"
                onClick={() => void confirmDelete()}
                disabled={!!deletingId}
                style={{
                  padding: "9px 16px",
                  background: "oklch(0.704 0.191 22.216 / 0.20)",
                  border: "1px solid oklch(0.78 0.16 22)",
                  borderRadius: 10,
                  color: "oklch(0.95 0.10 22)",
                  fontFamily: "var(--font-mono)",
                  fontSize: 11,
                  letterSpacing: "0.10em",
                  textTransform: "uppercase",
                  fontWeight: 600,
                  cursor: deletingId ? "wait" : "pointer",
                  opacity: deletingId ? 0.7 : 1,
                }}
              >
                {deletingId ? "Excluindo…" : "🗑 Excluir"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
