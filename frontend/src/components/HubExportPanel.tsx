import { useEffect, useState, type CSSProperties } from "react";
import {
  deleteHubToken, exportRunToHub, fetchHubPreview, fetchHubToken, saveHubToken,
  type HubPreview, type HubTokenStatus,
} from "../api/client";
import { useT } from "../i18n/useT";

const FIELD: CSSProperties = {
  background: "var(--vf-panel)", color: "var(--vf-text)", padding: 8,
  border: "1px solid var(--vf-panel-stroke)", borderRadius: 6,
};

/** Review a run snapshot and explicitly upload it with this profile's token. */
export function HubExportPanel({ runId }: { runId: string }) {
  const t = useT();
  const [open, setOpen] = useState(false);
  const [status, setStatus] = useState<HubTokenStatus | null>(null);
  const [preview, setPreview] = useState<HubPreview | null>(null);
  const [token, setToken] = useState("");
  const [repo, setRepo] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [url, setUrl] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    let alive = true;
    setBusy(true); setError(null); setPreview(null); setStatus(null); setUrl(null);
    Promise.all([fetchHubToken(), fetchHubPreview(runId)])
      .then(([tokenStatus, snapshot]) => {
        if (alive) { setStatus(tokenStatus); setPreview(snapshot); }
      })
      .catch((e: unknown) => { if (alive) setError(reason(e, t.hub.failed)); })
      .finally(() => { if (alive) setBusy(false); });
    return () => { alive = false; };
  }, [open, runId, t.hub.failed]);

  const credentials = async (remove: boolean) => {
    setBusy(true); setError(null);
    const secret = token;
    setToken("");
    try {
      const next = remove ? await deleteHubToken() : await saveHubToken(secret);
      setStatus(next);
    } catch (e) { setError(reason(e, t.hub.failed)); }
    finally { setBusy(false); }
  };

  const upload = async () => {
    if (!preview || !status?.configured) return;
    setBusy(true); setError(null); setUrl(null);
    try {
      const result = await exportRunToHub(runId, repo.trim(), preview.digest);
      const destination = new URL(result.url);
      if (destination.protocol !== "https:" || destination.hostname !== "huggingface.co") throw new Error();
      setUrl(destination.href);
    } catch (e) { setError(reason(e, t.hub.failed)); }
    finally { setBusy(false); }
  };

  if (!open) return <button type="button" onClick={() => setOpen(true)}>{t.hub.open}</button>;
  return <section aria-label={t.hub.title} style={{ border: "1px solid var(--vf-panel-stroke)", borderRadius: 10, padding: 16 }}>
    <h3>{t.hub.title}</h3>
    <button type="button" onClick={() => { setToken(""); setOpen(false); }} disabled={busy}>{t.hub.close}</button>
    {busy && !preview && <p role="status">{t.hub.loading}</p>}
    {status?.shared && <p role="alert">{t.hub.shared}</p>}
    {status && <div>
      {status.configured && <p>{t.hub.configured}</p>}
      <label>{t.hub.token} <input type="password" autoComplete="new-password" value={token}
        onChange={(e) => setToken(e.target.value)} disabled={busy} style={FIELD} /></label>
      <p>{t.hub.tokenHint}</p>
      <button type="button" disabled={busy || !token.trim()} onClick={() => void credentials(false)}>{t.hub.saveToken}</button>
      {status.configured && <button type="button" disabled={busy} onClick={() => void credentials(true)}>{t.hub.removeToken}</button>}
    </div>}
    {preview && <div>
      <h4>{t.hub.preview}</h4>
      <ul>{preview.files.map((file) => <li key={file}>{file}</li>)}</ul>
      <details><summary>{t.hub.card}</summary><pre style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>{preview.card}</pre></details>
      <label>{t.hub.repo} <input value={repo} onChange={(e) => { setRepo(e.target.value); setUrl(null); }} disabled={busy} style={FIELD} /></label>
      <p>{t.hub.confirm}</p>
      <p role="note">{t.hub.checkpointWarning}</p>
      <button type="button" disabled={busy || !status?.configured || !repo.trim()}
        onClick={() => void upload()}>{busy ? t.hub.busy : t.hub.send}</button>
    </div>}
    {error && <p role="alert">{error}</p>}
    {url && <a href={url} target="_blank" rel="noopener noreferrer">{t.hub.success}</a>}
  </section>;
}

/** The server's translated reason when it sent one; else the generic failure. */
function reason(error: unknown, fallback: string): string {
  return error instanceof Error && error.message ? error.message : fallback;
}
