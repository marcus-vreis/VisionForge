import { useState } from "react";
import { artifactUrl, previewAugment, type AugmentPreviewResponse } from "../api/client";
import { useT } from "../i18n/useT";

interface AugmentPreviewProps {
  baseDir: string;
  transforms: Record<string, unknown>;
}

/** Renders a strip of randomly-augmented variants of a sample image so the user
 *  can see the train-time augmentation effect (flip/rotation/jitter) before
 *  training. Mirrors the preprocessing preview, but for the random augmentations. */
export function AugmentPreview({ baseDir, transforms }: AugmentPreviewProps) {
  const t = useT();
  const [preview, setPreview] = useState<AugmentPreviewResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  const run = async () => {
    if (!baseDir.trim()) {
      setMsg(t.augmentPreview.needBaseDir);
      return;
    }
    setLoading(true);
    setMsg(null);
    setPreview(null);
    try {
      const resp = await previewAugment(baseDir, transforms ?? {}, { numVariants: 4 });
      if (resp.message) {
        setMsg(resp.message);
      } else {
        setPreview(resp);
      }
    } catch (e) {
      setMsg(e instanceof Error ? e.message : t.errors.previewFailed);
    } finally {
      setLoading(false);
    }
  };

  const thumb: React.CSSProperties = {
    width: 96,
    height: 96,
    objectFit: "cover",
    borderRadius: 8,
    border: "1px solid var(--vf-panel-stroke)",
    display: "block",
  };
  const cap: React.CSSProperties = {
    fontFamily: "var(--font-mono)",
    fontSize: 9,
    color: "var(--vf-text-muted)",
    textAlign: "center",
    marginTop: 4,
  };

  return (
    <div style={{ marginTop: 14 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
        <button
          type="button"
          onClick={() => void run()}
          disabled={loading}
          style={{
            padding: "8px 14px",
            background: "var(--accent-soft)",
            border: "1px solid var(--accent-vf)",
            borderRadius: 8,
            color: "var(--vf-text)",
            fontFamily: "var(--font-mono)",
            fontSize: 11,
            cursor: loading ? "wait" : "pointer",
            letterSpacing: "0.08em",
            textTransform: "uppercase",
            opacity: loading ? 0.6 : 1,
          }}
        >
          {loading ? t.augmentPreview.generating : t.augmentPreview.button}
        </button>
        {preview && preview.active.length > 0 && (
          <span
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: 10,
              color: "var(--vf-text-dim)",
            }}
          >
            {t.augmentPreview.active} {preview.active.join(" · ")}
          </span>
        )}
        {preview && preview.active.length === 0 && (
          <span
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: 10,
              color: "var(--vf-text-muted)",
            }}
          >
            {t.augmentPreview.noneActive}
          </span>
        )}
      </div>

      {msg && (
        <div
          style={{
            marginTop: 8,
            fontFamily: "var(--font-mono)",
            fontSize: 11,
            color: "var(--vf-text-muted)",
          }}
        >
          {msg}
        </div>
      )}

      {preview && (
        <div
          style={{
            marginTop: 12,
            display: "flex",
            gap: 14,
            flexWrap: "wrap",
            alignItems: "flex-start",
          }}
        >
          <div>
            <img src={artifactUrl(preview.original)} alt={t.augmentPreview.original} style={thumb} />
            <div style={cap}>{t.augmentPreview.original}</div>
          </div>
          {preview.variants.map((v, i) => (
            <div key={v}>
              <img src={artifactUrl(v)} alt={t.augmentPreview.variant(i + 1)} style={thumb} />
              <div style={cap}>{t.augmentPreview.variant(i + 1)}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
