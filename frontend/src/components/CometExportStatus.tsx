import { useState } from "react";
import { retryCometExport, type CometExportStatus as Status } from "../api/client";
import { useT } from "../i18n/useT";

/** The run's Comet copy: a link when it is there, the reason and a retry when not. */
export function CometExportStatus({ runId, status, onUpdated }: {
  runId: string; status?: Status | null; onUpdated: (status: Status) => void;
}) {
  const t = useT();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  if (!status) return null;
  const link = safeLink(status.url);
  if (status.state === "finished" && link) {
    return <a href={link} target="_blank" rel="noopener noreferrer">{t.comet.open}</a>;
  }
  if (status.state !== "failed" && status.state !== "disabled") return null;
  const retry = async () => {
    setBusy(true); setError(null);
    try { onUpdated(await retryCometExport(runId)); }
    catch (e) { setError(e instanceof Error && e.message ? e.message : t.comet.retryFailed); }
    finally { setBusy(false); }
  };
  return <div role="status">
    <p>{reasonText(status, t.comet)}</p>
    <button type="button" disabled={busy} onClick={() => void retry()}>{busy ? t.comet.retrying : t.comet.retry}</button>
    {error && <p role="alert">{error}</p>}
  </div>;
}

/** Only an https address becomes a link; anything else is not shown. */
function safeLink(url: string | null | undefined): string | null {
  if (!url) return null;
  try {
    const parsed = new URL(url);
    return parsed.protocol === "https:" ? parsed.href : null;
  } catch {
    return null;
  }
}

function reasonText(
  status: Status,
  text: { notInstalled: string; tokenMissing: string; notPrivate: string; busy: string; warning: string },
): string {
  if (status.state === "disabled") return text.notInstalled;
  if (status.error === "CometTokenMissingError") return text.tokenMissing;
  if (status.error === "CometPrivateProjectError") return text.notPrivate;
  if (status.error === "CometBusyError") return text.busy;
  return `${text.warning}${status.error ? `: ${status.error}` : ""}`;
}
