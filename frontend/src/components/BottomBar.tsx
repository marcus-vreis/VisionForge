interface BottomBarProps {
  onHistory: () => void;
  onTrain: () => void;
  disabled: boolean;
  historyCount: number;
  device: "cuda" | "cpu";
  setDevice: (d: "cuda" | "cpu") => void;
  gpuName: string;
  isRunning: boolean;
}

/** Fixed bottom action bar with History, Treinar, and device indicator. */
export function BottomBar({
  onHistory,
  onTrain,
  disabled,
  historyCount,
  device,
  setDevice,
  gpuName,
  isRunning,
}: BottomBarProps) {
  return (
    <div
      style={{
        position: "fixed",
        bottom: 24,
        left: 24,
        right: 24,
        maxWidth: 1280,
        margin: "0 auto",
        zIndex: 5,
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        gap: 12,
        padding: "12px 14px",
        background: "rgba(8,10,14,0.78)",
        backdropFilter: "blur(20px)",
        WebkitBackdropFilter: "blur(20px)",
        border: "1px solid var(--vf-panel-stroke)",
        borderRadius: 16,
        boxShadow: "0 30px 80px rgba(0,0,0,0.5)",
      }}
    >
      {/* History button */}
      <button
        type="button"
        onClick={onHistory}
        style={{
          display: "flex",
          alignItems: "center",
          gap: 10,
          padding: "10px 16px",
          background: "rgba(255,255,255,0.025)",
          border: "1px solid var(--vf-panel-stroke)",
          borderRadius: 10,
          color: "var(--vf-text-dim)",
          fontFamily: "var(--font-mono)",
          fontSize: 12,
          letterSpacing: "0.08em",
          textTransform: "uppercase",
          cursor: "pointer",
        }}
      >
        <span style={{ fontSize: 14 }}>⤺</span>
        history
        {historyCount > 0 && (
          <span
            style={{
              marginLeft: 4,
              padding: "2px 7px",
              background: "var(--accent-soft)",
              color: "var(--accent-vf)",
              borderRadius: 999,
              fontSize: 10,
              letterSpacing: "0.04em",
            }}
          >
            {historyCount}
          </span>
        )}
      </button>

      {/* Treinar button */}
      <button
        type="button"
        onClick={onTrain}
        disabled={disabled}
        style={{
          flex: "0 0 auto",
          position: "relative",
          padding: "14px 38px",
          background:
            "linear-gradient(180deg, var(--accent-soft) 0%, rgba(8,10,14,0.4) 100%)",
          border: "1px solid var(--accent-vf)",
          borderRadius: 12,
          color: "var(--vf-text)",
          fontFamily: "var(--font-mono)",
          fontSize: 13,
          fontWeight: 600,
          letterSpacing: "0.18em",
          textTransform: "uppercase",
          boxShadow:
            "inset 0 0 18px var(--accent-glow), 0 0 30px var(--accent-soft)",
          overflow: "hidden",
          cursor: disabled ? "not-allowed" : "pointer",
          opacity: disabled ? 0.55 : 1,
        }}
      >
        <span
          style={{
            position: "absolute",
            inset: 0,
            background:
              "linear-gradient(90deg, transparent, var(--accent-soft), transparent)",
            backgroundSize: "200% 100%",
            animation: "shimmer 3.6s linear infinite",
            opacity: 0.6,
          }}
        />
        <span style={{ position: "relative" }}>
          {isRunning ? "Executando…" : "▶ Treinar"}
        </span>
      </button>

      {/* Device indicator */}
      <button
        type="button"
        onClick={() => setDevice(device === "cuda" ? "cpu" : "cuda")}
        title="Toggle compute device (cosmetic)"
        style={{
          display: "flex",
          alignItems: "center",
          gap: 12,
          padding: "10px 16px",
          background:
            device === "cuda"
              ? "oklch(0.78 0.18 150 / 0.10)"
              : "rgba(255,255,255,0.025)",
          border: "1px solid",
          borderColor:
            device === "cuda"
              ? "oklch(0.78 0.18 150 / 0.5)"
              : "var(--vf-panel-stroke)",
          borderRadius: 999,
          color: "var(--vf-text)",
          fontFamily: "var(--font-mono)",
          fontSize: 11,
          letterSpacing: "0.12em",
          textTransform: "uppercase",
          cursor: "pointer",
        }}
      >
        <span
          style={{
            width: 8,
            height: 8,
            borderRadius: "50%",
            background:
              device === "cuda"
                ? "oklch(0.78 0.18 150)"
                : "oklch(0.74 0.10 70)",
            boxShadow:
              device === "cuda"
                ? "0 0 12px oklch(0.78 0.18 150 / 0.8)"
                : "none",
            animation: "pulse-dot 1.6s ease-in-out infinite",
            flexShrink: 0,
          }}
        />
        <span style={{ color: "var(--vf-text-dim)" }}>usando</span>
        <span
          style={{
            color:
              device === "cuda"
                ? "oklch(0.85 0.16 150)"
                : "oklch(0.85 0.10 70)",
            fontWeight: 600,
          }}
        >
          {device === "cuda" ? "CUDA" : "CPU"}
        </span>
        {device === "cuda" && (
          <span
            style={{
              color: "var(--vf-text-muted)",
              fontSize: 10,
              letterSpacing: "0.08em",
            }}
          >
            · {gpuName}
          </span>
        )}
      </button>
    </div>
  );
}
