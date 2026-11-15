import { useState } from "react";
import { retryWandbExport, type WandbExportStatus as Status } from "../api/client";
import { useT } from "../i18n/useT";

/** The run's W&B copy: a link when it is there, the reason and a retry when not. */
export function WandbExportStatus({ runId, status, onUpdated }: {
  runId: string; status?: Status | null; onUpdated: (status: Status) => void;
}) {
  const t = useT();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  if (!status) return null;
  const link = safeLink(status.url);
  if (status.state === "finished" && link) {
    return <a href={link} target="_blank" rel="noopener noreferrer">{t.wandb.open}</a>;
  }
  if (status.state !== "failed" && status.state !== "disabled") return null;
  const retry = async () => {
    setBusy(true); setError(null);
    try { onUpdated(await retryWandbExport(runId)); }
    catch (e) { setError(e instanceof Error && e.message ? e.message : t.wandb.retryFailed); }
    finally { setBusy(false); }
  };
  return <div role="status">
    <p>{reasonText(status, t.wandb)}</p>
    <button type="button" disabled={busy} onClick={() => void retry()}>{busy ? t.wandb.retrying : t.wandb.retry}</button>
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
  text: { notInstalled: string; tokenMissing: string; notPrivate: string; warning: string },
): string {
  if (status.state === "disabled") return text.notInstalled;
  if (status.error === "WandbTokenMissingError") return text.tokenMissing;
  if (status.error === "WandbPrivateProjectError") return text.notPrivate;
  return `${text.warning}${status.error ? `: ${status.error}` : ""}`;
}
