import { useEffect } from "react";

interface LightboxProps {
  src: string;
  alt?: string;
  caption?: string;
  onClose: () => void;
}

/** Full-screen image overlay — Esc / click outside to close. */
export function Lightbox({ src, alt, caption, onClose }: LightboxProps) {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onClose]);

  // Caption + close button live as absolute overlays so they never steal
  // vertical space from the image. Previously the image was constrained to
  // 85vh in a column flex layout with caption underneath, which clipped tall
  // plots (confusion matrices, ROC curves) on shorter viewports.
  return (
    <div
      onClick={onClose}
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 300,
        background: "rgba(2,3,5,0.94)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        overflow: "auto",
        padding: 24,
        cursor: "zoom-out",
      }}
    >
      <img
        src={src}
        alt={alt ?? ""}
        onClick={(e) => e.stopPropagation()}
        style={{
          maxWidth: "calc(100vw - 48px)",
          maxHeight: "calc(100vh - 48px)",
          width: "auto",
          height: "auto",
          objectFit: "contain",
          borderRadius: 12,
          boxShadow: "0 30px 90px rgba(0,0,0,0.7)",
          background: "#0a0c10",
          cursor: "default",
          display: "block",
        }}
      />
      {caption && (
        <div
          onClick={(e) => e.stopPropagation()}
          style={{
            position: "fixed",
            left: "50%",
            bottom: 18,
            transform: "translateX(-50%)",
            maxWidth: "min(92vw, 720px)",
            padding: "8px 14px",
            background: "rgba(0,0,0,0.65)",
            backdropFilter: "blur(6px)",
            WebkitBackdropFilter: "blur(6px)",
            border: "1px solid var(--vf-panel-stroke)",
            borderRadius: 10,
            fontFamily: "var(--font-mono)",
            fontSize: 12,
            color: "var(--vf-text-dim)",
            wordBreak: "break-all",
            textAlign: "center",
            pointerEvents: "auto",
          }}
        >
          {caption}
        </div>
      )}
      <button
        type="button"
        onClick={onClose}
        title="Fechar (Esc)"
        style={{
          position: "fixed",
          top: 22,
          right: 22,
          width: 38,
          height: 38,
          borderRadius: "50%",
          background: "rgba(255,255,255,0.06)",
          border: "1px solid var(--vf-panel-stroke)",
          color: "var(--vf-text)",
          fontSize: 22,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          cursor: "pointer",
        }}
      >
        ×
      </button>
    </div>
  );
}
