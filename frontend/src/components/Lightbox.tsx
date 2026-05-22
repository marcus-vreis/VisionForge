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

  return (
    <div
      onClick={onClose}
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 300,
        background: "rgba(2,3,5,0.92)",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        padding: 32,
        cursor: "zoom-out",
      }}
    >
      <img
        src={src}
        alt={alt ?? ""}
        onClick={(e) => e.stopPropagation()}
        style={{
          maxWidth: "92vw",
          maxHeight: "85vh",
          objectFit: "contain",
          borderRadius: 12,
          boxShadow: "0 30px 90px rgba(0,0,0,0.7)",
          background: "#0a0c10",
          cursor: "default",
        }}
      />
      {caption && (
        <div
          style={{
            marginTop: 18,
            padding: "8px 14px",
            background: "rgba(0,0,0,0.5)",
            border: "1px solid var(--vf-panel-stroke)",
            borderRadius: 10,
            fontFamily: "var(--font-mono)",
            fontSize: 12,
            color: "var(--vf-text-dim)",
            maxWidth: "92vw",
            wordBreak: "break-all",
            textAlign: "center",
          }}
        >
          {caption}
        </div>
      )}
      <button
        type="button"
        onClick={onClose}
        style={{
          position: "absolute",
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
