import { DatasetDownloadCard } from "./DatasetDownloadCard";

interface DatasetsOverlayProps {
  open: boolean;
  onClose: () => void;
}

/** Datasets as a surface of its own, reachable from the bottom bar.
 *
 * The download form used to render at the bottom of every task panel — five
 * copies of one global action, each pushing the panel it belonged to further
 * down. A dataset is not owned by a task: you fetch it once and then point
 * whichever panel you like at the folder. So it lives here, next to History,
 * and the panels are back to being only about their experiment.
 */
export function DatasetsOverlay({ open, onClose }: DatasetsOverlayProps) {
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
          width: "min(760px, 100%)",
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
              // datasets
            </div>
            <div
              style={{
                fontSize: 22,
                fontWeight: 600,
                letterSpacing: "-0.01em",
                color: "var(--vf-text)",
              }}
            >
              Obter dados
            </div>
            <div
              style={{
                marginTop: 6,
                fontFamily: "var(--font-mono)",
                fontSize: 11,
                color: "var(--vf-text-dim)",
                lineHeight: 1.6,
              }}
            >
              Baixe uma vez para uma pasta local; depois aponte o campo de
              dataset de qualquer task para ela.
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

        <div style={{ flex: 1, overflowY: "auto", padding: "16px 24px 24px" }}>
          <DatasetDownloadCard collapsible={false} />
        </div>
      </div>
    </div>
  );
}
