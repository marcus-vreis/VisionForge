import { useCallback, useEffect, useState } from "react";
import { fetchRuns } from "../api/client";
import type { RunSummary } from "../types/run";
import { Button } from "./ui/button";

const TASK_ACCENT: Record<string, string> = {
  binary:       "oklch(0.74 0.18 22)",
  multiclass:   "oklch(0.74 0.18 22)",
  detection:    "oklch(0.78 0.18 150)",
  regression:   "oklch(0.74 0.16 240)",
  segmentation: "oklch(0.74 0.18 305)",
};

function accent(task: string): string {
  return TASK_ACCENT[task] ?? TASK_ACCENT.binary;
}

function MiniCurve({ values, color }: { values: number[]; color: string }) {
  if (values.length < 2) return null;
  const w = 220, h = 56, pad = 4;
  const max = Math.max(...values);
  const min = Math.min(...values);
  const span = max - min || 1;
  const pts = values.map((v, i) => [
    pad + (i / (values.length - 1)) * (w - pad * 2),
    pad + (1 - (v - min) / span) * (h - pad * 2),
  ]);
  const d = pts.map(([x, y], i) => `${i === 0 ? "M" : "L"} ${x!.toFixed(1)} ${y!.toFixed(1)}`).join(" ");
  const dFill = `${d} L ${pts[pts.length - 1]![0]} ${h - pad} L ${pts[0]![0]} ${h - pad} Z`;
  const gradId = `mg-${color.replace(/[^a-z0-9]/gi, "")}`;
  return (
    <svg width={w} height={h} style={{ display: "block" }}>
      <defs>
        <linearGradient id={gradId} x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity={0.4} />
          <stop offset="100%" stopColor={color} stopOpacity={0} />
        </linearGradient>
      </defs>
      <path d={dFill} fill={`url(#${gradId})`} />
      <path d={d} fill="none" stroke={color} strokeWidth="1.4" strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  );
}

function formatDate(iso: string): string {
  return new Date(iso).toLocaleString(undefined, {
    month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
  });
}

function primaryMetric(run: RunSummary): { label: string; value: string } {
  const m = run.final_metrics;
  for (const key of ["f1", "accuracy", "auc", "mse", "map"]) {
    if (key in m) return { label: key.toUpperCase(), value: m[key]!.toFixed(4) };
  }
  const first = Object.entries(m)[0];
  if (first) return { label: first[0], value: first[1].toFixed(4) };
  return { label: "—", value: "—" };
}

interface RunHistoryOverlayProps {
  open: boolean;
  onClose: () => void;
  onResume: (config: Record<string, unknown>) => void;
}

export function RunHistoryOverlay({ open, onClose, onResume }: RunHistoryOverlayProps) {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    fetchRuns()
      .then((data) => {
        setRuns(data);
        if (data.length > 0) setSelectedId(data[0]!.run_id);
      })
      .catch(() => setError("Failed to load run history."))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (open) load();
  }, [open, load]);

  if (!open) return null;

  const selected = runs.find((r) => r.run_id === selectedId) ?? null;

  return (
    <div
      onClick={onClose}
      className="fixed inset-0 z-50 flex items-center justify-center p-6"
      style={{ background: "rgba(4,5,7,0.75)" }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="w-full max-h-[92vh] overflow-hidden rounded-2xl border border-border/50"
        style={{
          maxWidth: 1100,
          display: "grid",
          gridTemplateColumns: "340px 1fr",
          background: "rgba(12,14,18,0.96)",
          boxShadow: "0 50px 120px rgba(0,0,0,0.7)",
        }}
      >
        {/* LEFT: run list */}
        <div className="flex flex-col border-r border-border/50 overflow-hidden" style={{ background: "rgba(8,10,14,0.6)" }}>
          <div className="px-5 py-4 border-b border-border/50">
            <div className="text-[10px] tracking-[0.22em] text-muted-foreground font-mono uppercase mb-1">
              // training history
            </div>
            <div className="text-lg font-semibold tracking-tight">Recent Runs</div>
            <div className="text-xs text-muted-foreground font-mono mt-1">
              {loading ? "Loading…" : `${runs.length} run${runs.length !== 1 ? "s" : ""}`}
            </div>
          </div>

          <div className="flex-1 overflow-y-auto p-2">
            {error && (
              <div className="p-3 text-sm text-destructive">{error}</div>
            )}
            {!loading && runs.length === 0 && !error && (
              <div className="p-4 text-sm text-muted-foreground text-center">No runs found.</div>
            )}
            {runs.map((run) => {
              const ac = accent(run.task);
              const active = run.run_id === selectedId;
              const metric = primaryMetric(run);
              return (
                <button
                  key={run.run_id}
                  onClick={() => setSelectedId(run.run_id)}
                  className="w-full text-left rounded-xl px-3 py-3 mb-1.5 relative border transition-colors"
                  style={{
                    background: active ? "rgba(255,255,255,0.04)" : "transparent",
                    borderColor: active ? "rgba(255,255,255,0.08)" : "transparent",
                  }}
                >
                  {active && (
                    <span
                      className="absolute left-0 top-3.5 bottom-3.5 w-0.5 rounded-full"
                      style={{ background: ac, boxShadow: `0 0 8px ${ac}` }}
                    />
                  )}
                  <div className="flex items-center gap-2 text-[10px] tracking-[0.16em] uppercase font-mono text-muted-foreground mb-1">
                    <span className="w-1.5 h-1.5 rounded-full shrink-0" style={{ background: ac, boxShadow: `0 0 6px ${ac}` }} />
                    {run.task} · {run.model_arch}
                    {run.status === "failed" && (
                      <span className="ml-auto text-destructive text-[9px]">FAILED</span>
                    )}
                  </div>
                  <div className="font-mono text-sm text-foreground mb-1.5 truncate">{run.experiment_name}</div>
                  <div className="flex justify-between font-mono text-[11px] text-muted-foreground">
                    <span>{run.finished_at ? formatDate(run.finished_at) : "running…"}</span>
                    <span style={{ color: ac }}>{metric.value}</span>
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        {/* RIGHT: detail */}
        <div className="flex flex-col overflow-y-auto">
          {selected ? (
            <RunDetail run={selected} onClose={onClose} onResume={onResume} />
          ) : (
            <div className="flex-1 flex items-center justify-center text-muted-foreground text-sm">
              {loading ? "Loading…" : "Select a run"}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function RunDetail({
  run,
  onClose,
  onResume,
}: {
  run: RunSummary;
  onClose: () => void;
  onResume: (config: Record<string, unknown>) => void;
}) {
  const ac = accent(run.task);
  const metric = primaryMetric(run);

  const curveValues = Object.entries(run.final_metrics)
    .filter(([k]) => k.startsWith("loss"))
    .map(([, v]) => v);

  return (
    <div className="p-6 flex flex-col gap-5 min-h-full">
      {/* Header */}
      <div className="flex items-start gap-4">
        <div className="flex-1">
          <div className="flex items-center gap-2.5 mb-2">
            <span
              className="text-[10px] tracking-[0.2em] uppercase font-mono px-2 py-1 rounded-full border"
              style={{ color: ac, borderColor: ac, background: `color-mix(in srgb, ${ac} 12%, transparent)` }}
            >
              {run.task}
            </span>
            <span className="text-[10px] font-mono text-muted-foreground">{run.run_id}</span>
            {run.finished_at && (
              <>
                <span className="text-muted-foreground/40">·</span>
                <span className="text-[10px] font-mono text-muted-foreground">{formatDate(run.finished_at)}</span>
              </>
            )}
          </div>
          <div className="text-xl font-semibold font-mono tracking-tight">{run.experiment_name}</div>
        </div>
        <button
          onClick={onClose}
          className="w-9 h-9 rounded-full border border-border/50 flex items-center justify-center text-xl leading-none text-muted-foreground hover:text-foreground transition-colors shrink-0"
        >
          ×
        </button>
      </div>

      {/* Metric cards */}
      <div className="grid grid-cols-4 gap-3">
        <MetricCard label={metric.label} value={metric.value} accent={ac} highlight />
        <MetricCard label="Model" value={run.model_arch} accent={ac} />
        <MetricCard label="Epochs" value={String(run.epochs_completed)} accent={ac} />
        <MetricCard label="Status" value={run.status} accent={ac} />
      </div>

      {/* All metrics */}
      {Object.keys(run.final_metrics).length > 0 && (
        <div className="rounded-xl border border-border/50 p-4" style={{ background: "rgba(8,10,14,0.5)" }}>
          <div className="text-[10px] tracking-[0.22em] uppercase font-mono text-muted-foreground mb-3">
            Final Metrics
          </div>
          <div className="grid grid-cols-3 gap-2">
            {Object.entries(run.final_metrics).map(([k, v]) => (
              <div key={k} className="flex justify-between font-mono text-xs">
                <span className="text-muted-foreground">{k}</span>
                <span className="text-foreground">{v.toFixed(4)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Mini loss curve if data available */}
      {curveValues.length >= 2 && (
        <div className="rounded-xl border border-border/50 p-4" style={{ background: "rgba(8,10,14,0.5)" }}>
          <div className="text-[10px] tracking-[0.22em] uppercase font-mono text-muted-foreground mb-3">
            Loss
          </div>
          <MiniCurve values={curveValues} color={ac} />
        </div>
      )}

      {/* Actions */}
      <div className="flex gap-2 mt-auto pt-2">
        <Button
          variant="outline"
          size="sm"
          onClick={() => onResume({ name: run.experiment_name, task: run.task })}
        >
          Resume config
        </Button>
      </div>
    </div>
  );
}

function MetricCard({
  label,
  value,
  accent: ac,
  highlight,
}: {
  label: string;
  value: string;
  accent: string;
  highlight?: boolean;
}) {
  return (
    <div
      className="p-3.5 rounded-xl border border-border/50 relative overflow-hidden"
      style={{
        background: highlight
          ? `linear-gradient(180deg, color-mix(in srgb, ${ac} 14%, transparent) 0%, rgba(12,14,18,0.5) 100%)`
          : "rgba(12,14,18,0.55)",
      }}
    >
      {highlight && (
        <span
          className="absolute top-2.5 right-2.5 w-1.5 h-1.5 rounded-full"
          style={{ background: ac, boxShadow: `0 0 10px ${ac}` }}
        />
      )}
      <div className="text-[10px] tracking-[0.18em] uppercase font-mono text-muted-foreground">{label}</div>
      <div
        className="text-lg font-mono font-semibold mt-1 truncate"
        style={{ color: highlight ? ac : undefined }}
      >
        {value}
      </div>
    </div>
  );
}
