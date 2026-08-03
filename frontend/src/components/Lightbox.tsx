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

  // Three rows that each take only the height they need, with the image row
  // absorbing everything left over (ADR-078). Neither of the two obvious
  // layouts works: a fixed 85vh image clips tall plots on short viewports, and
  // floating the caption and the close button over the image — the previous
  // fix — hides the strip a matplotlib figure puts its x-axis and legend in.
  // Reserving the space instead means nothing is covered and nothing is cut.
  return (
    <div
      onClick={onClose}
      // Marks the topmost layer: the history overlay's Esc handler checks for
      // this and stands down while an image is open, so one Esc closes one
      // level (ADR-078).
      data-lightbox=""
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 300,
        background: "rgba(2,3,5,0.94)",
        display: "flex",
        flexDirection: "column",
        padding: 16,
        gap: 10,
        cursor: "zoom-out",
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "flex-end",
          flex: "0 0 auto",
        }}
      >
        <button
          type="button"
          onClick={onClose}
          title="Fechar (Esc)"
          style={{
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

      {/* min-height: 0 is what lets this row actually shrink; without it a flex
          item refuses to go below its content size and the image overflows. */}
      <div
        style={{
          flex: "1 1 auto",
          minHeight: 0,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <img
          src={src}
          alt={alt ?? ""}
          onClick={(e) => e.stopPropagation()}
          style={{
            maxWidth: "100%",
            maxHeight: "100%",
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
      </div>

      {caption && (
        <div
          onClick={(e) => e.stopPropagation()}
          style={{
            flex: "0 0 auto",
            alignSelf: "center",
            maxWidth: "min(92vw, 720px)",
            padding: "8px 14px",
            background: "rgba(0,0,0,0.65)",
            border: "1px solid var(--vf-panel-stroke)",
            borderRadius: 10,
            fontFamily: "var(--font-mono)",
            fontSize: 12,
            color: "var(--vf-text-dim)",
            wordBreak: "break-all",
            textAlign: "center",
          }}
        >
          {caption}
        </div>
      )}
    </div>
  );
}
