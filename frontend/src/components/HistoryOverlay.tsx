import { useCallback, useEffect, useState } from "react";
import { deleteRun, fetchRuns } from "../api/client";
import type { RunSummary } from "../types/run";
import { CompareRunsPanel } from "./CompareRunsPanel";
import { MenuSelect } from "./controls";
import { RunDetailPanel } from "./RunDetailPanel";

interface HistoryOverlayProps {
  onClose: () => void;
  onCountChange?: (count: number) => void;
  /** Task family the app is currently on. Opening the history from the
   * Classification panel means you want *its* runs — landing on "Todos" and
   * making you click again is a step nobody wants. Falls back to "all" when
   * the active task has no runs yet, so the sheet is never empty on open. */
  initialTask?: string;
}

/** Inline caption preceding a filter control. */
function FilterLabel({ children }: { children: React.ReactNode }) {
  return (
    <span
      style={{
        fontFamily: "var(--font-mono)",
        fontSize: 9,
        letterSpacing: "0.16em",
        textTransform: "uppercase",
        color: "var(--vf-text-muted)",
      }}
    >
      {children}
    </span>
  );
}

/** One filter dimension as a row of chips that wraps.
 *
 * Chips, not a dropdown: every option and the current one are readable without
 * opening anything. The clipping that made the original row unusable came from
 * it being a single non-wrapping line — `flexWrap` fixes that directly, and a
 * long list grows downward instead of disappearing off the edge.
 *
 * The label sits on its own line so the chips always start from the same
 * column, however long the caption is.
 */
function FilterChips({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: { value: string; label: string }[];
  onChange: (v: string) => void;
}) {
  const entries = [{ value: "all", label: "todos" }, ...options];
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      <FilterLabel>{label}</FilterLabel>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
        {entries.map((o) => {
          const active = o.value === value;
          return (
            <button
              key={o.value}
              type="button"
              onClick={() => onChange(o.value)}
              style={{
                padding: "5px 11px",
                background: active ? "var(--accent-soft)" : "rgba(255,255,255,0.04)",
                border: `1px solid ${active ? "var(--accent-vf)" : "var(--vf-panel-stroke)"}`,
                borderRadius: 999,
                fontFamily: "var(--font-mono)",
                fontSize: 10,
                letterSpacing: "0.10em",
                textTransform: "uppercase",
                color: active ? "var(--vf-text)" : "var(--vf-text-dim)",
                cursor: "pointer",
                whiteSpace: "nowrap",
              }}
            >
              {o.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}

/** Color accent per task type, mirroring the VisionForge oklch palette. */
const TASK_ACCENT: Record<string, string> = {
  classification: "oklch(0.74 0.18 22)",
  detection: "oklch(0.78 0.18 150)",
  regression: "oklch(0.74 0.16 240)",
  segmentation: "oklch(0.74 0.18 305)",
  anomaly: "oklch(0.80 0.15 75)",
};

/** Tab label per task family; custom tasks (ADR-058) keep their own key. */
const FAMILY_LABELS: Record<string, string> = {
  classification: "Classificação",
  detection: "Detecção",
  regression: "Regressão",
  segmentation: "Segmentação",
  anomaly: "Anomalia",
};

/** Order the family tabs the way the app's own task bar orders them, so the
 * history reads like the rest of the GUI instead of alphabetically. */
const FAMILY_ORDER = [
  "classification",
  "detection",
  "regression",
  "segmentation",
  "anomaly",
];

/** Map a run's `task` onto the family it belongs to.
 *
 * `run.task` is not the family: classification runs record their *problem*
 * type (`binary`, `multiclass`, `multilabel`) because that is what the
 * classification config's `task` field means, while the standalone tasks
 * record the family itself. Grouping on the raw value split classification
 * into a "BINARY" and a "MULTICLASS" tab, which is not a task anyone chose.
 */
function taskFamily(task: string): string {
  if (task.startsWith("custom:")) return task;
  if (FAMILY_LABELS[task] !== undefined && task !== "classification") return task;
  return "classification";
}

function familyLabel(family: string): string {
  if (family.startsWith("custom:")) return family.slice("custom:".length);
  return FAMILY_LABELS[family] ?? family;
}

/** One history tab per task, each scoped to that task's runs.
 *
 * Replaces the task chip row: a researcher looks for "that detection run",
 * not for a run among all tasks, so the task is navigation rather than a
 * filter competing for space with status, block and sort.
 */
function TaskTabs({
  tasks,
  counts,
  active,
  onSelect,
  total,
}: {
  tasks: string[];
  counts: Record<string, number>;
  active: string;
  onSelect: (task: string) => void;
  total: number;
}) {
  const entries: { key: string; label: string; count: number }[] = [
    { key: "all", label: "Todos", count: total },
    ...tasks.map((t) => ({ key: t, label: familyLabel(t), count: counts[t] ?? 0 })),
  ];
  return (
    <div
      style={{
        display: "flex",
        gap: 2,
        overflowX: "auto",
        scrollbarWidth: "none",
        borderBottom: "1px solid var(--vf-panel-stroke)",
        padding: "0 24px",
        flexShrink: 0,
      }}
    >
      {entries.map((e) => {
        const on = e.key === active;
        const accent = e.key === "all" ? "var(--vf-text-dim)" : TASK_ACCENT[e.key] ?? "var(--vf-text-muted)";
        return (
          <button
            key={e.key}
            type="button"
            onClick={() => onSelect(e.key)}
            style={{
              padding: "10px 14px",
              whiteSpace: "nowrap",
              flexShrink: 0,
              background: "transparent",
              border: "none",
              borderBottom: `2px solid ${on ? accent : "transparent"}`,
              color: on ? "var(--vf-text)" : "var(--vf-text-dim)",
              fontFamily: "var(--font-mono)",
              fontSize: 11,
              letterSpacing: "0.10em",
              textTransform: "uppercase",
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              gap: 7,
            }}
          >
            {e.label}
            <span
              style={{
                padding: "1px 6px",
                borderRadius: 999,
                background: on ? "rgba(255,255,255,0.10)" : "rgba(255,255,255,0.04)",
                fontSize: 10,
                color: "var(--vf-text-dim)",
              }}
            >
              {e.count}
            </span>
          </button>
        );
      })}
    </div>
  );
}

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

/** Metric keys shown on each card when present, per task. Mirrors the
 * backend `_SUMMARY_METRIC_KEYS` projection in routes.py. */
const METRIC_KEYS_BY_TASK: Record<string, string[]> = {
  classification: ["accuracy", "f1", "val_loss"],
  detection: ["map50", "map50_95"],
};

/** Human-readable labels for metric keys that aren't self-explanatory. */
const METRIC_LABELS: Record<string, string> = {
  map50: "mAP@50",
  map50_95: "mAP@50-95",
};

/** Ordering options for the run list. */
type SortKey = "recent" | "oldest" | "epochs";

const SORT_LABELS: Record<SortKey, string> = {
  recent: "mais recente",
  oldest: "mais antigo",
  epochs: "mais épocas",
};

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
  // Keyed by family: a classification run records `binary`/`multiclass`, which
  // has no accent of its own — the card used to fall back to grey for the most
  // common task in the list.
  const accent = TASK_ACCENT[taskFamily(run.task)] ?? "var(--vf-text-muted)";
  const dot = statusColor(run.status);
  // A researcher-defined task (custom:<key>, ADR-058) declares its own metric
  // names, so there is no fixed key list — the backend already projected the
  // ones worth showing, and they render under their real names.
  const metricKeys = run.task.startsWith("custom:")
    ? Object.keys(run.final_metrics)
    : (METRIC_KEYS_BY_TASK[run.task] ?? METRIC_KEYS_BY_TASK.classification);
  const shownMetrics = metricKeys.filter(
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
        {run.dataset_name && (
          <span
            title={run.dataset_root ?? run.dataset_name}
            style={{
              padding: "2px 8px",
              background: "oklch(0.70 0.12 250 / 0.14)",
              border: "1px solid oklch(0.70 0.12 250 / 0.45)",
              borderRadius: 999,
              fontFamily: "var(--font-mono)",
              fontSize: 10,
              color: "oklch(0.86 0.11 250)",
              letterSpacing: "0.10em",
              // A long dataset name must not push the other pills out of the
              // row; the row wraps, but the pill itself has to stay bounded.
              maxWidth: 180,
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
            }}
          >
            🗂 {run.dataset_name}
          </span>
        )}
        {run.resumable && (
          <span
            title={
              run.configured_epochs
                ? `Parou na época ${run.epochs_completed} de ${run.configured_epochs} — dá para continuar`
                : "Parou antes do fim — dá para continuar"
            }
            style={{
              padding: "2px 8px",
              background: "oklch(0.80 0.16 85 / 0.14)",
              border: "1px solid oklch(0.80 0.16 85 / 0.45)",
              borderRadius: 999,
              fontFamily: "var(--font-mono)",
              fontSize: 10,
              color: "oklch(0.90 0.15 85)",
              letterSpacing: "0.10em",
              textTransform: "uppercase",
            }}
          >
            ⏸ {run.epochs_completed}
            {run.configured_epochs ? `/${run.configured_epochs}` : ""}
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
                {METRIC_LABELS[k] ?? k}
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
export function HistoryOverlay({
  onClose,
  onCountChange,
  initialTask,
}: HistoryOverlayProps) {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  // Mounted means opening, and opening always fetches.
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [selectMode, setSelectMode] = useState(false);
  const [selection, setSelection] = useState<string[]>([]);
  const [compareActiveIds, setCompareActiveIds] = useState<string[] | null>(null);
  const [query, setQuery] = useState("");
  const [taskFilter, setTaskFilter] = useState<string>("all");
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const [blockFilter, setBlockFilter] = useState<string>("all");
  // The raw `run.task` within a family — `binary` vs `multiclass` inside
  // Classificação. Only meaningful where the family has more than one.
  const [subtypeFilter, setSubtypeFilter] = useState<string>("all");
  const [sortBy, setSortBy] = useState<SortKey>("recent");
  const [pendingDeletes, setPendingDeletes] = useState<RunSummary[] | null>(null);
  const [deletingIds, setDeletingIds] = useState<string[]>([]);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  // A stray click outside the sheet used to throw away the whole navigation, so
  // the backdrop and Esc both step back exactly one level: gráfico → treinamento
  // → histórico → gui. The × in the header is the deliberate "close it all".
  // Declared before the `!open` early return so the Esc effect can depend on it.
  const stepBack = useCallback(() => {
    if (selectedRunId) {
      setSelectedRunId(null);
      return;
    }
    if (compareActiveIds) {
      setCompareActiveIds(null);
      return;
    }
    onClose();
  }, [selectedRunId, compareActiveIds, onClose]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      // The lightbox is on top and owns Esc while it is open; it stops there.
      if (e.key === "Escape" && !document.querySelector("[data-lightbox]")) {
        stepBack();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [stepBack]);

  useEffect(() => {
    // No resetting here: this component is mounted when the sheet opens, so the
    // initial state *is* the reset. It used to stay mounted and re-clear twelve
    // pieces of state on the way in, which is a cascade of renders to reach the
    // values a fresh mount already has.
    fetchRuns()
      .then((data) => {
        setRuns(data);
        onCountChange?.(data.length);
        // Only land on the active task's tab if that tab will actually exist.
        if (initialTask && data.some((r) => taskFamily(r.task) === initialTask)) {
          setTaskFilter(initialTask);
        }
      })
      .catch((e: unknown) => {
        const msg =
          e instanceof Error ? e.message : "Erro ao carregar histórico.";
        setError(msg);
      })
      .finally(() => setLoading(false));
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const toggleSelect = (runId: string) => {
    setSelection((prev) =>
      prev.includes(runId) ? prev.filter((id) => id !== runId) : [...prev, runId],
    );
  };

  /** Delete every run in `pendingDeletes`, one request at a time.
   *
   * Sequential rather than parallel: each call is an `rmtree` on the same
   * disk, and a partial failure has to name the runs that survived, which a
   * racing `Promise.all` cannot do cleanly. Runs that succeeded are dropped
   * from the list even when a later one fails, so the view never claims a
   * deleted run still exists. The overlay stays open either way.
   */
  const confirmDelete = async () => {
    if (!pendingDeletes || pendingDeletes.length === 0) return;
    const targets = pendingDeletes.map((r) => r.run_id);
    setDeletingIds(targets);
    setDeleteError(null);

    const deleted: string[] = [];
    const failures: string[] = [];
    for (const id of targets) {
      try {
        await deleteRun(id);
        deleted.push(id);
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : "erro desconhecido";
        failures.push(`${id}: ${msg}`);
      }
    }

    if (deleted.length > 0) {
      const next = runs.filter((r) => !deleted.includes(r.run_id));
      setRuns(next);
      onCountChange?.(next.length);
      if (selectedRunId && deleted.includes(selectedRunId)) setSelectedRunId(null);
      setSelection((prev) => prev.filter((rid) => !deleted.includes(rid)));
    }
    setDeletingIds([]);

    if (failures.length > 0) {
      setDeleteError(
        `${failures.length} de ${targets.length} não puderam ser excluídos:\n${failures.join("\n")}`,
      );
      // Keep the dialog open on the ones that survived so the message has a
      // subject; dismissing it is the researcher's call.
      setPendingDeletes(pendingDeletes.filter((r) => !deleted.includes(r.run_id)));
      return;
    }
    setPendingDeletes(null);
  };

  // Client-side filter + sort — keeps the list responsive even with hundreds of
  // runs. Search is case-insensitive and matches experiment name, arch or run_id.
  const filteredRuns = (() => {
    if (runs.length === 0) return runs;
    const q = query.trim().toLowerCase();
    const matches = runs.filter((r) => {
      if (taskFilter !== "all" && taskFamily(r.task) !== taskFilter) return false;
      if (subtypeFilter !== "all" && r.task !== subtypeFilter) return false;
      if (statusFilter !== "all" && r.status !== statusFilter) return false;
      if (blockFilter !== "all" && (r.block ?? "classification") !== blockFilter)
        return false;
      if (q === "") return true;
      return (
        r.experiment_name.toLowerCase().includes(q) ||
        r.model_arch.toLowerCase().includes(q) ||
        r.run_id.toLowerCase().includes(q)
      );
    });
    const sorted = [...matches];
    if (sortBy === "oldest") {
      sorted.sort((a, b) => a.started_at.localeCompare(b.started_at));
    } else if (sortBy === "epochs") {
      sorted.sort((a, b) => b.epochs_completed - a.epochs_completed);
    } else {
      sorted.sort((a, b) => b.started_at.localeCompare(a.started_at));
    }
    return sorted;
  })();

  const taskCounts = runs.reduce<Record<string, number>>((acc, r) => {
    const f = taskFamily(r.task);
    acc[f] = (acc[f] ?? 0) + 1;
    return acc;
  }, {});
  // Built-in families in the GUI's own order, then the researcher's custom
  // tasks alphabetically after them.
  const presentFamilies = new Set(Object.keys(taskCounts));
  const availableTasks = [
    ...FAMILY_ORDER.filter((f) => presentFamilies.has(f)),
    ...Array.from(presentFamilies).filter((f) => f.startsWith("custom:")).sort(),
  ];

  // Status and block options are derived from the runs of the *active tab*, so
  // a tab never offers a filter that would empty its own list.
  const tabRuns =
    taskFilter === "all"
      ? runs
      : runs.filter((r) => taskFamily(r.task) === taskFilter);
  const availableStatuses = Array.from(new Set(tabRuns.map((r) => r.status))).sort();
  const availableBlocks = Array.from(
    new Set(tabRuns.map((r) => r.block ?? "classification")),
  ).sort();
  // Classification records `binary`/`multiclass`/`multilabel` as its task, so
  // inside that tab the raw value is a real second dimension to filter by.
  const availableSubtypes = Array.from(new Set(tabRuns.map((r) => r.task))).sort();
  const activeFilterCount =
    (query !== "" ? 1 : 0) +
    (statusFilter !== "all" ? 1 : 0) +
    (subtypeFilter !== "all" ? 1 : 0) +
    (blockFilter !== "all" ? 1 : 0);

  const selectedRuns = runs.filter((r) => selection.includes(r.run_id));
  const busyDeleting = deletingIds.length > 0;


  return (
    <div
      onClick={stepBack}
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
              : // Wide enough for the family tab row not to need scrolling at
                // the five built-ins plus a custom task.
                "min(760px, 100%)",
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
            {/* One selection mode drives both actions. Separate "compare
                mode" and "delete mode" toggles would make the researcher
                declare intent before picking runs, which is backwards: you
                pick the runs, then decide what to do with them. */}
            {!selectedRunId && !compareActiveIds && runs.length > 0 && (
              <button
                type="button"
                onClick={() => {
                  setSelectMode((m) => !m);
                  if (selectMode) setSelection([]);
                }}
                style={{
                  padding: "8px 14px",
                  background: selectMode ? "oklch(0.78 0.18 150 / 0.18)" : "rgba(255,255,255,0.04)",
                  border: `1px solid ${selectMode ? "oklch(0.78 0.18 150)" : "var(--vf-panel-stroke)"}`,
                  borderRadius: 10,
                  color: selectMode ? "oklch(0.88 0.16 150)" : "var(--vf-text-dim)",
                  fontFamily: "var(--font-mono)",
                  fontSize: 11,
                  letterSpacing: "0.10em",
                  textTransform: "uppercase",
                  cursor: "pointer",
                }}
              >
                {selectMode ? "Cancelar seleção" : "✓ Selecionar"}
              </button>
            )}
            {selectMode && selection.length >= 2 && (
              <button
                type="button"
                onClick={() => setCompareActiveIds(selection)}
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
                ↔ Comparar {selection.length}
              </button>
            )}
            {selectMode && selection.length >= 1 && (
              <button
                type="button"
                onClick={() => {
                  setDeleteError(null);
                  setPendingDeletes(selectedRuns);
                }}
                title={`Excluir ${selection.length} run(s) permanentemente`}
                style={{
                  padding: "8px 14px",
                  background: "oklch(0.704 0.191 22.216 / 0.18)",
                  border: "1px solid oklch(0.78 0.16 22)",
                  borderRadius: 10,
                  color: "oklch(0.92 0.12 22)",
                  fontFamily: "var(--font-mono)",
                  fontSize: 11,
                  letterSpacing: "0.10em",
                  textTransform: "uppercase",
                  cursor: "pointer",
                  fontWeight: 600,
                }}
              >
                🗑 Excluir {selection.length}
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

        {/* One tab per task — the researcher navigates by task, then filters
            inside it. Hidden while a run detail or a comparison is open. */}
        {!selectedRunId &&
          !compareActiveIds &&
          !loading &&
          error === null &&
          availableTasks.length > 1 && (
            <TaskTabs
              tasks={availableTasks}
              counts={taskCounts}
              active={taskFilter}
              onSelect={(t) => {
                setTaskFilter(t);
                // All three are scoped to the tab; carrying them across would
                // show an empty list under a filter the new tab cannot satisfy.
                setStatusFilter("all");
                setBlockFilter("all");
                setSubtypeFilter("all");
              }}
              total={runs.length}
            />
          )}

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
                setSelectMode(false);
                setSelection([]);
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

          {/* Selection-mode tip — states what each count unlocks, because the
              two actions have different thresholds (delete 1+, compare 2+). */}
          {!selectedRunId && !compareActiveIds && selectMode && (
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
              {selection.length === 0
                ? "Modo seleção — marque os runs que quer excluir ou comparar."
                : `${selection.length} selecionado(s) — 🗑 exclui; ↔ compara a partir de 2.`}
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
              {activeFilterCount > 0 && (
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

          {/* Refinements *inside* the active tab. Each dimension is its own
              wrapping row, so a long option list grows downward instead of
              running off the edge, and only dimensions that actually vary in
              this tab are shown — a single-valued filter filters nothing. */}
          {!selectedRunId && !compareActiveIds && !loading && error === null && runs.length > 0 && (
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                gap: 10,
                marginBottom: 14,
              }}
            >
              {availableSubtypes.length > 1 && (
                <FilterChips
                  label="tipo"
                  value={subtypeFilter}
                  options={availableSubtypes.map((s) => ({
                    value: s,
                    label: s.startsWith("custom:") ? s.slice(7) : s,
                  }))}
                  onChange={setSubtypeFilter}
                />
              )}
              {availableBlocks.length > 1 && (
                <FilterChips
                  label="bloco"
                  value={blockFilter}
                  options={availableBlocks.map((b) => ({ value: b, label: b }))}
                  onChange={setBlockFilter}
                />
              )}
              {availableStatuses.length > 1 && (
                <FilterChips
                  label="status"
                  value={statusFilter}
                  options={availableStatuses.map((s) => ({ value: s, label: s }))}
                  onChange={setStatusFilter}
                />
              )}
              <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <FilterLabel>ordenar</FilterLabel>
                <MenuSelect
                  value={sortBy}
                  onChange={(v) => setSortBy(v as SortKey)}
                  options={(Object.keys(SORT_LABELS) as SortKey[]).map((k) => ({
                    value: k,
                    label: SORT_LABELS[k],
                  }))}
                  minWidth={150}
                />
              </div>
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
                    selectable={selectMode}
                    selected={selection.includes(run.run_id)}
                    onToggleSelect={() => toggleSelect(run.run_id)}
                    onDelete={() => {
                      setDeleteError(null);
                      setPendingDeletes([run]);
                    }}
                    deleting={deletingIds.includes(run.run_id)}
                  />
                ))
              )}
            </div>
          )}
        </div>
      </div>

      {/* Delete confirmation modal — a layer above the history sheet. It sits
          inside the overlay's backdrop, whose onClick closes the history, so
          every click in here must stop propagating: without it, confirming or
          cancelling a delete also closed the whole history. */}
      {pendingDeletes && pendingDeletes.length > 0 && (
        <div
          onClick={(e) => {
            e.stopPropagation();
            if (e.target === e.currentTarget && deletingIds.length === 0) {
              setPendingDeletes(null);
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
              // excluir {pendingDeletes.length === 1 ? "run" : `${pendingDeletes.length} runs`}{" "}
              permanentemente
            </div>
            {/* Every run is named, however many: "excluir 12 runs" without the
                list is a destructive action taken on trust. Scrolls past ~6. */}
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                gap: 4,
                maxHeight: 168,
                overflowY: "auto",
              }}
            >
              {pendingDeletes.map((r) => (
                <div
                  key={r.run_id}
                  style={{
                    fontFamily: "var(--font-mono)",
                    fontSize: 12,
                    color: "var(--vf-text)",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                  }}
                >
                  {r.experiment_name}
                  <span style={{ color: "var(--vf-text-muted)", marginLeft: 6 }}>
                    · {r.model_arch}
                  </span>
                </div>
              ))}
            </div>
            <div
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: 11,
                color: "var(--vf-text-dim)",
                lineHeight: 1.6,
              }}
            >
              {pendingDeletes.length === 1 ? "A pasta do run" : "As pastas dos runs"},
              checkpoints e todos os plots/relatórios serão removidos do disco.
              Esta ação é irreversível.
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
                  setPendingDeletes(null);
                  setDeleteError(null);
                }}
                disabled={busyDeleting}
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
                  cursor: busyDeleting ? "not-allowed" : "pointer",
                  opacity: busyDeleting ? 0.5 : 1,
                }}
              >
                {deleteError ? "Fechar" : "Cancelar"}
              </button>
              <button
                type="button"
                onClick={() => void confirmDelete()}
                disabled={busyDeleting}
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
                  cursor: busyDeleting ? "wait" : "pointer",
                  opacity: busyDeleting ? 0.7 : 1,
                }}
              >
                {busyDeleting
                  ? "Excluindo…"
                  : `🗑 Excluir${pendingDeletes.length > 1 ? ` ${pendingDeletes.length}` : ""}`}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
