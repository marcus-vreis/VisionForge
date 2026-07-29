import { DeviceSelector, type DeviceSelection } from "./DeviceSelector";

interface BottomBarProps {
  onHistory: () => void;
  onDatasets: () => void;
  onTrain: () => void;
  disabled: boolean;
  historyCount: number;
  selection: DeviceSelection;
  onSelectionChange: (next: DeviceSelection) => void;
  isRunning: boolean;
  /** True when training is running but the overlay was minimized; clicking
   * the Treinar button should reopen the live view instead of starting a
   * second run. */
  trainingMinimized?: boolean;
  onReopenTraining?: () => void;
  /** What the button will actually run, from the panel's strategy selector —
   * so "Treinar" never silently starts a plain run while Sweep is selected. */
  trainLabel?: string;
}

/** Fixed bottom action bar with History, Treinar, and device selector. */
export function BottomBar({
  onHistory,
  onDatasets,
  onTrain,
  disabled,
  historyCount,
  selection,
  onSelectionChange,
  isRunning,
  trainingMinimized = false,
  onReopenTraining,
  trainLabel,
}: BottomBarProps) {
  // While a run is minimized, the central button reopens the overlay rather
  // than triggering a new training (a duplicate run would race with the
  // current one). The label flips to make the action obvious.
  const handleCenterClick = () => {
    if (trainingMinimized && onReopenTraining) {
      onReopenTraining();
      return;
    }
    onTrain();
  };

  const centerLabel = trainingMinimized
    ? "🔬 abrir treino"
    : isRunning
      ? "Executando…"
      : (trainLabel ?? "▶ Treinar");
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
        // Fixed bar over scrolling content re-blurs every frame; the surface is
        // already ~80% opaque, so a smaller radius is visually identical and
        // roughly halves the per-frame fill cost.
        background: "rgba(8,10,14,0.82)",
        backdropFilter: "blur(10px)",
        WebkitBackdropFilter: "blur(10px)",
        border: "1px solid var(--vf-panel-stroke)",
        borderRadius: 16,
        boxShadow: "0 30px 80px rgba(0,0,0,0.5)",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
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

      <button
        type="button"
        onClick={onDatasets}
        title="Baixar um dataset para uma pasta local"
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
        <span style={{ fontSize: 14 }}>⤓</span>
        datasets
      </button>
      </div>

      <button
        type="button"
        onClick={handleCenterClick}
        disabled={disabled && !trainingMinimized}
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
          cursor: disabled && !trainingMinimized ? "not-allowed" : "pointer",
          opacity: disabled && !trainingMinimized ? 0.55 : 1,
        }}
      >
        <span
          style={{
            position: "absolute",
            inset: 0,
            background:
              "linear-gradient(90deg, transparent, var(--accent-soft), transparent)",
            backgroundSize: "200% 100%",
            // Only animate while a run is active. Idle, this span sits inside
            // the blurred bottom bar and re-triggered its backdrop-filter every
            // frame, contributing to the UI-wide jank.
            animation: isRunning ? "shimmer 3.6s linear infinite" : "none",
            opacity: 0.6,
          }}
        />
        <span style={{ position: "relative" }}>{centerLabel}</span>
      </button>

      <DeviceSelector
        selection={selection}
        onChange={onSelectionChange}
        variant="bottom"
      />
    </div>
  );
}
