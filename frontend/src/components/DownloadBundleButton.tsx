import { useState } from "react";
import { Package } from "lucide-react";
import { downloadRunBundle } from "../api/client";
import { useT } from "../i18n/useT";
import { ActionButton } from "./ui/ActionButton";

/** Download an experiment bundle and keep failures visible beside the action.
 *
 * `label` is a shorter visible text for a row already titled "download"; the
 * button's name stays the full action, which the short text is part of. */
export function DownloadBundleButton({ runId, label }: { runId: string; label?: string }) {
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

  const shortLabel = !loading && label !== undefined && label !== t.evidence.bundle;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      <ActionButton
        icon={<Package />}
        onClick={() => void download()}
        disabled={loading}
        aria-busy={loading}
        aria-label={shortLabel ? t.evidence.bundle : undefined}
      >
        {loading ? t.evidence.bundling : shortLabel ? label : t.evidence.bundle}
      </ActionButton>
      {error && (
        <div role="alert" style={{ fontSize: 13, color: "oklch(calc(0.80 - var(--lt-flip, 0) * 0.3) 0.17 22)" }}>
          {t.evidence.bundleFailed} {error}
        </div>
      )}
    </div>
  );
}
