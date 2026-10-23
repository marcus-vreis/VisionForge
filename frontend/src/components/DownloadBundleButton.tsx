import { useState } from "react";
import { downloadRunBundle } from "../api/client";
import { useT } from "../i18n/useT";

/** Download an experiment bundle and keep failures visible beside the action. */
export function DownloadBundleButton({ runId }: { runId: string }) {
  const t = useT();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function download() {
    setLoading(true);
    setError(null);
    try {
      await downloadRunBundle(runId);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : t.evidence.bundleFailed);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <button type="button" onClick={() => void download()} disabled={loading} aria-busy={loading}
        style={{ padding: "6px 12px", background: "var(--accent-soft)", border: "1px solid var(--accent-vf)", borderRadius: 8, color: "var(--vf-text)", cursor: loading ? "wait" : "pointer" }}>
        {loading ? t.evidence.bundling : t.evidence.bundle}
      </button>
      {error && <div role="alert">{t.evidence.bundleFailed} {error}</div>}
    </div>
  );
}
