import { useState } from "react";
import { retryMLflowExport, type MLflowTrackingStatus } from "../api/client";
import { useT } from "../i18n/useT";

/** Show diagnostics and explicit export retries. */
export function MLflowExportStatus({ runId, tracking, onUpdated }: {
  runId: string; tracking?: MLflowTrackingStatus | null;
  onUpdated: (tracking: MLflowTrackingStatus) => void;
}) {
  const t = useT();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  if (!tracking || !["failed", "disabled"].includes(tracking.state)) return null;
  const retry = async () => {
    setBusy(true); setError(null);
    try { onUpdated(await retryMLflowExport(runId)); }
    catch (e) { setError(e instanceof Error ? e.message : t.mlflow.retryFailed); }
    finally { setBusy(false); }
  };
  return <div role="status">
    <p>
      {tracking.state === "disabled"
        ? t.mlflow.notInstalled
        : `${t.mlflow.warning}${tracking.error ? `: ${tracking.error}` : ""}`}
    </p>
    <button type="button" disabled={busy} onClick={() => void retry()}>{busy ? t.mlflow.retrying : t.mlflow.retry}</button>
    {error && <p role="alert">{error}</p>}
  </div>;
}
