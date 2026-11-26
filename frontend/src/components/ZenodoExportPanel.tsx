import { useEffect, useState, type CSSProperties } from "react";
import {
  createZenodoDraft, deleteZenodoToken, fetchZenodoPreview, fetchZenodoToken, saveZenodoToken,
  type HubTokenStatus, type ZenodoPreview, type ZenodoSite,
} from "../api/client";
import { formatBytes } from "../lib/dataset-identity";
import { useI18n } from "../i18n/useT";
import { ActionButton } from "./ui/ActionButton";
import { ZenodoIcon } from "./ui/BrandIcons";
import { TokenStorageNote } from "./TokenStorageNote";

const FIELD: CSSProperties = {
  background: "var(--vf-panel)", color: "var(--vf-text)", padding: 8,
  border: "1px solid var(--vf-panel-stroke)", borderRadius: 6,
};
const HOSTS: Record<ZenodoSite, string> = { zenodo: "zenodo.org", sandbox: "sandbox.zenodo.org" };
// The note the server adds when it rewrites run.json; the page words it itself.
const PATHS_NOTE = "run.json";

/** Review a run's bundle and put it in an unpublished Zenodo draft (ADR-174). */
export function ZenodoExportPanel({ runId }: { runId: string }) {
  const { t, locale } = useI18n();
  const z = t.zenodo;
  const [open, setOpen] = useState(false);
  const [site, setSite] = useState<ZenodoSite>("zenodo");
  const [status, setStatus] = useState<HubTokenStatus | null>(null);
  const [preview, setPreview] = useState<ZenodoPreview | null>(null);
  const [token, setToken] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [url, setUrl] = useState<string | null>(null);

  // Each effect below starts a fresh load when its deps change while open: the
  // state it shows is reset while rendering, before that fetch runs.
  const previewKey = open ? JSON.stringify([runId, z.failed]) : null;
  const [seenPreviewKey, setSeenPreviewKey] = useState<string | null>(null);
  if (previewKey !== seenPreviewKey) {
    setSeenPreviewKey(previewKey);
    if (previewKey !== null) { setBusy(true); setError(null); setPreview(null); setUrl(null); }
  }
  const tokenKey = open ? JSON.stringify([site, z.failed]) : null;
  const [seenTokenKey, setSeenTokenKey] = useState<string | null>(null);
  if (tokenKey !== seenTokenKey) {
    setSeenTokenKey(tokenKey);
    if (tokenKey !== null) setStatus(null);
  }

  useEffect(() => {
    if (!open) return;
    let alive = true;
    fetchZenodoPreview(runId)
      .then((snapshot) => { if (alive) setPreview(snapshot); })
      .catch((e: unknown) => { if (alive) setError(reason(e, z.failed)); })
      .finally(() => { if (alive) setBusy(false); });
    return () => { alive = false; };
  }, [open, runId, z.failed]);

  useEffect(() => {
    if (!open) return;
    let alive = true;
    fetchZenodoToken(site)
      .then((next) => { if (alive) setStatus(next); })
      .catch((e: unknown) => { if (alive) setError(reason(e, z.failed)); });
    return () => { alive = false; };
  }, [open, site, z.failed]);

  const credentials = async (remove: boolean) => {
    setBusy(true); setError(null);
    const secret = token;
    setToken("");
    try {
      setStatus(remove ? await deleteZenodoToken(site) : await saveZenodoToken(site, secret));
    } catch (e) { setError(reason(e, z.failed)); }
    finally { setBusy(false); }
  };

  const upload = async () => {
    if (!preview || !status?.configured) return;
    setBusy(true); setError(null); setUrl(null);
    try {
      const draft = await createZenodoDraft(runId, preview.digest, site);
      const destination = new URL(draft.url);
      if (destination.protocol !== "https:" || destination.hostname !== HOSTS[site]) throw new Error();
      setUrl(destination.href);
    } catch (e) { setError(reason(e, z.failed)); }
    finally { setBusy(false); }
  };

  const choose = (next: ZenodoSite) => { setSite(next); setToken(""); setUrl(null); setError(null); };

  // Closed: the service's mark and name; the button's name says what opening does.
  if (!open) return <ActionButton icon={<ZenodoIcon />} aria-label={z.open} title={z.open}
    onClick={() => setOpen(true)}>{z.service}</ActionButton>;
  const left = preview?.notes.filter((note) => note.artifact !== PATHS_NOTE) ?? [];
  return <section aria-label={z.title} style={{ border: "1px solid var(--vf-panel-stroke)", borderRadius: 10, padding: 16 }}>
    <h3>{z.title}</h3>
    <button type="button" onClick={() => { setToken(""); setOpen(false); }} disabled={busy}>{z.close}</button>
    <p>{z.intro}</p>
    <fieldset disabled={busy} style={{ border: "none", padding: 0 }}>
      <legend>{z.site}</legend>
      {(["zenodo", "sandbox"] as const).map((option) => <label key={option} style={{ display: "block" }}>
        <input type="radio" name={`zenodo-site-${runId}`} checked={site === option} onChange={() => choose(option)} />
        {" "}{option === "sandbox" ? z.sandbox : z.production}
      </label>)}
    </fieldset>
    {site === "sandbox" && <p>{z.sandboxHint}</p>}
    {status?.shared && <p role="alert">{z.shared}</p>}
    <TokenStorageNote />
    {status && <div>
      {status.configured && <p>{z.configured}</p>}
      <label>{z.token} <input type="password" autoComplete="new-password" value={token}
        onChange={(e) => setToken(e.target.value)} disabled={busy} style={FIELD} /></label>
      <p>{z.tokenHint}</p>
      <button type="button" disabled={busy || !token.trim()} onClick={() => void credentials(false)}>{z.saveToken}</button>
      {status.configured && <button type="button" disabled={busy} onClick={() => void credentials(true)}>{z.removeToken}</button>}
    </div>}
    {busy && !preview && <p role="status">{z.loading}</p>}
    {preview && <div>
      <h4>{z.preview}</h4>
      <p>{z.file(preview.name, formatBytes(preview.size, locale))}</p>
      <ul>{preview.files.map((file) => <li key={file}>{file}</li>)}</ul>
      <p>{z.pathsNote}</p>
      {left.length > 0 && <details><summary>{z.leftOut(left.length)}</summary>
        <ul>{left.map((note, i) => <li key={`${note.artifact}-${i}`}>{note.artifact}: {note.reason}</li>)}</ul>
      </details>}
      <p role="note">{z.checkpointWarning}</p>
      <p>{z.confirm}</p>
      <button type="button" disabled={busy || !status?.configured} onClick={() => void upload()}>
        {busy ? z.busy : z.send}
      </button>
    </div>}
    {error && <p role="alert">{error}</p>}
    {url && <p>
      <a href={url} target="_blank" rel="noopener noreferrer">{z.success}</a>{" "}{z.next}
    </p>}
  </section>;
}

/** The server's translated reason when it sent one; else the generic failure. */
function reason(error: unknown, fallback: string): string {
  return error instanceof Error && error.message ? error.message : fallback;
}
