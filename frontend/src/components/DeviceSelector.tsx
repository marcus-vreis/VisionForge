import { useEffect, useRef, useState } from "react";
import { fetchDeviceInfo, type DeviceInfoResponse, type GPUInfo } from "../api/client";

export interface DeviceSelection {
  kind: "cpu" | "cuda" | "multi_cuda";
  gpu_ids: number[] | null;
}

interface DeviceSelectorProps {
  selection: DeviceSelection;
  onChange: (next: DeviceSelection) => void;
  variant?: "header" | "bottom";
}

function formatMem(mb: number): string {
  if (mb <= 0) return "";
  if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`;
  return `${mb} MB`;
}

function labelForSelection(
  sel: DeviceSelection,
  info: DeviceInfoResponse | null,
): string {
  if (sel.kind === "cpu") return "CPU";
  if (!info || info.gpus.length === 0) return "CUDA";
  if (sel.kind === "multi_cuda") {
    const ids = sel.gpu_ids ?? info.gpus.map((g) => g.index);
    return `MULTI · ${ids.length} GPUs`;
  }
  const id = sel.gpu_ids?.[0] ?? 0;
  const g = info.gpus.find((x) => x.index === id);
  return g ? `GPU ${id}` : "CUDA";
}

/** Real device picker — fetches GPUs from the backend and stores the choice. */
export function DeviceSelector({ selection, onChange, variant = "bottom" }: DeviceSelectorProps) {
  const [info, setInfo] = useState<DeviceInfoResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState(false);
  const wrapRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    let alive = true;
    fetchDeviceInfo()
      .then((d) => {
        if (!alive) return;
        setInfo(d);
        setLoading(false);
        // Auto-correct invalid initial selection.
        if (!d.cuda_available && selection.kind !== "cpu") {
          onChange({ kind: "cpu", gpu_ids: null });
        }
        if (selection.kind === "multi_cuda" && d.gpus.length < 2) {
          onChange({ kind: d.cuda_available ? "cuda" : "cpu", gpu_ids: null });
        }
      })
      .catch((e: unknown) => {
        if (!alive) return;
        setError(e instanceof Error ? e.message : "Falha ao consultar dispositivos.");
        setLoading(false);
      });
    return () => {
      alive = false;
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!open) return;
    const onClick = (e: MouseEvent) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    window.addEventListener("mousedown", onClick);
    return () => window.removeEventListener("mousedown", onClick);
  }, [open]);

  const cudaActive = selection.kind !== "cpu" && info?.cuda_available;
  const label = labelForSelection(selection, info);

  return (
    <div ref={wrapRef} style={{ position: "relative" }}>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        title={
          loading
            ? "Carregando dispositivos…"
            : error
              ? error
              : info?.cuda_available
                ? `CUDA ${info.cuda_version ?? ""} · ${info.gpus.length} GPU(s)`
                : "CUDA indisponível — somente CPU"
        }
        style={{
          display: "flex",
          alignItems: "center",
          gap: 10,
          padding: variant === "header" ? "8px 14px" : "10px 16px",
          background: cudaActive
            ? "oklch(0.78 0.18 150 / 0.10)"
            : "rgba(255,255,255,0.03)",
          border: "1px solid",
          borderColor: cudaActive
            ? "oklch(0.78 0.18 150 / 0.5)"
            : "var(--vf-panel-stroke)",
          borderRadius: 999,
          fontFamily: "var(--font-mono)",
          fontSize: 11,
          letterSpacing: "0.12em",
          textTransform: "uppercase",
          cursor: "pointer",
          color: "var(--vf-text)",
        }}
      >
        <span
          style={{
            width: 8,
            height: 8,
            borderRadius: "50%",
            background: cudaActive ? "oklch(0.78 0.18 150)" : "oklch(0.74 0.10 70)",
            boxShadow: cudaActive ? "0 0 12px oklch(0.78 0.18 150 / 0.8)" : "none",
            animation: "pulse-dot 1.6s ease-in-out infinite",
            flexShrink: 0,
          }}
        />
        <span style={{ color: "var(--vf-text-dim)" }}>usando</span>
        <span
          style={{
            color: cudaActive ? "oklch(0.85 0.16 150)" : "oklch(0.85 0.10 70)",
            fontWeight: 600,
          }}
        >
          {label}
        </span>
        <span style={{ fontSize: 9, color: "var(--vf-text-muted)" }}>▾</span>
      </button>

      {open && (
        <div
          style={{
            position: "absolute",
            bottom: variant === "bottom" ? "calc(100% + 10px)" : undefined,
            top: variant === "header" ? "calc(100% + 10px)" : undefined,
            right: 0,
            minWidth: 280,
            background: "rgba(12,14,18,0.96)",
            border: "1px solid var(--vf-panel-stroke)",
            borderRadius: 14,
            boxShadow: "0 30px 80px rgba(0,0,0,0.6)",
            padding: 10,
            zIndex: 50,
            display: "flex",
            flexDirection: "column",
            gap: 4,
          }}
        >
          <div
            style={{
              padding: "8px 12px",
              fontFamily: "var(--font-mono)",
              fontSize: 10,
              letterSpacing: "0.16em",
              textTransform: "uppercase",
              color: "var(--vf-text-muted)",
            }}
          >
            {loading
              ? "carregando…"
              : error
                ? `erro: ${error}`
                : info?.cuda_available
                  ? `cuda ${info.cuda_version ?? ""} · ${info.gpus.length} gpu(s)`
                  : "cuda não detectado"}
          </div>

          <DeviceOption
            active={selection.kind === "cpu"}
            title={info?.cpu_name || "CPU"}
            subtitle="Treinar usando processador"
            onClick={() => {
              onChange({ kind: "cpu", gpu_ids: null });
              setOpen(false);
            }}
          />

          {info?.gpus.map((g: GPUInfo) => (
            <DeviceOption
              key={g.index}
              active={
                selection.kind === "cuda" &&
                (selection.gpu_ids?.[0] ?? 0) === g.index
              }
              title={`GPU ${g.index} · ${g.name}`}
              subtitle={[formatMem(g.total_memory_mb), g.compute_capability && `CC ${g.compute_capability}`]
                .filter(Boolean)
                .join(" · ")}
              onClick={() => {
                onChange({ kind: "cuda", gpu_ids: [g.index] });
                setOpen(false);
              }}
            />
          ))}

          {info && info.gpus.length >= 2 && (
            <DeviceOption
              active={selection.kind === "multi_cuda"}
              title={`Multi-GPU (${info.gpus.length} GPUs)`}
              subtitle="DataParallel"
              onClick={() => {
                onChange({
                  kind: "multi_cuda",
                  gpu_ids: info.gpus.map((g) => g.index),
                });
                setOpen(false);
              }}
            />
          )}
        </div>
      )}
    </div>
  );
}

interface DeviceOptionProps {
  active: boolean;
  title: string;
  subtitle?: string;
  onClick: () => void;
}

function DeviceOption({ active, title, subtitle, onClick }: DeviceOptionProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      style={{
        textAlign: "left",
        padding: "10px 12px",
        background: active ? "var(--accent-soft)" : "transparent",
        border: "1px solid",
        borderColor: active ? "var(--accent-vf)" : "transparent",
        borderRadius: 10,
        cursor: "pointer",
        color: "var(--vf-text)",
        display: "flex",
        flexDirection: "column",
        gap: 2,
        fontFamily: "var(--font-mono)",
      }}
    >
      <span style={{ fontSize: 12, fontWeight: 600 }}>{title}</span>
      {subtitle && (
        <span style={{ fontSize: 10, color: "var(--vf-text-muted)" }}>{subtitle}</span>
      )}
    </button>
  );
}
