interface HistoryOverlayProps {
  open: boolean;
  onClose: () => void;
}

/** History overlay — shows empty state (run history not yet exposed by backend). */
export function HistoryOverlay({ open, onClose }: HistoryOverlayProps) {
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
          width: "min(640px, 100%)",
          background: "rgba(12,14,18,0.95)",
          border: "1px solid var(--vf-panel-stroke)",
          borderRadius: 18,
          boxShadow: "0 50px 120px rgba(0,0,0,0.7)",
          overflow: "hidden",
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
            }}
          >
            ×
          </button>
        </div>

        {/* Empty state */}
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
      </div>
    </div>
  );
}
