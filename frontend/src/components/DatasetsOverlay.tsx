import { useState } from "react";
import { useT } from "../i18n/useT";
import { DatasetDownloadCard } from "./DatasetDownloadCard";
import { DatasetExplorer } from "./DatasetExplorer";

type Tab = "get" | "explore";

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
  const t = useT();
  const [tab, setTab] = useState<Tab>("get");
  if (!open) return null;

  return (
    <div
      onClick={onClose}
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 100,
        background: "rgb(var(--lt-surface, 4 5 7) / 0.72)",
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
          width: tab === "explore" ? "min(980px, 100%)" : "min(760px, 100%)",
          maxHeight: "85vh",
          background: "rgb(var(--lt-surface, 12 14 18) / 0.95)",
          border: "1px solid var(--vf-panel-stroke)",
          borderRadius: 18,
          boxShadow: "0 50px 120px rgb(var(--lt-shade, 0 0 0) / 0.7)",
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
              {t.datasetsOverlay.kicker}
            </div>
            <div
              style={{
                fontSize: 22,
                fontWeight: 600,
                letterSpacing: "-0.01em",
                color: "var(--vf-text)",
              }}
            >
              {t.datasetsOverlay.title}
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
              {t.datasetsOverlay.description}
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
              background: "rgb(var(--lt-ink, 255 255 255) / 0.03)",
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

        <div
          role="tablist"
          aria-label={t.datasetsOverlay.tabsLabel}
          style={{
            display: "flex",
            gap: 6,
            padding: "10px 24px 0",
            flexShrink: 0,
          }}
        >
          {(["get", "explore"] as const).map((id) => (
            <button
              key={id}
              type="button"
              role="tab"
              aria-selected={tab === id}
              onClick={() => setTab(id)}
              style={{
                padding: "7px 14px",
                borderRadius: 8,
                border: `1px solid ${tab === id ? "var(--accent-vf)" : "var(--vf-panel-stroke)"}`,
                background: tab === id ? "var(--accent-soft)" : "transparent",
                color: tab === id ? "var(--vf-text)" : "var(--vf-text-dim)",
                fontFamily: "var(--font-mono)",
                fontSize: 11,
                letterSpacing: "0.10em",
                textTransform: "uppercase",
                cursor: "pointer",
              }}
            >
              {id === "get" ? t.datasetsOverlay.tabGet : t.datasetsOverlay.tabExplore}
            </button>
          ))}
        </div>

        <div style={{ flex: 1, overflowY: "auto", padding: "16px 24px 24px" }}>
          {tab === "get" ? <DatasetDownloadCard collapsible={false} /> : <DatasetExplorer />}
        </div>
      </div>
    </div>
  );
}
