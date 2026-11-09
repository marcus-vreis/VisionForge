import { useEffect, useState, type CSSProperties } from "react";
import {
  commitRunToGitHub, deleteGitHubToken, fetchGitHubPreview, fetchGitHubToken, saveGitHubToken,
  type GitHubPreview, type HubTokenStatus,
} from "../api/client";
import { formatBytes } from "../lib/dataset-identity";
import { useI18n } from "../i18n/useT";

// The only host a commit link may point at.
const GITHUB_HOST = /^github\.com$/;
const FIELD: CSSProperties = {
  background: "var(--vf-panel)", color: "var(--vf-text)", padding: 8,
  border: "1px solid var(--vf-panel-stroke)", borderRadius: 6,
};

/** Review a run's report and commit it, without weights, to a private repository (ADR-175). */
export function GitHubExportPanel({ runId }: { runId: string }) {
  const { t, locale } = useI18n();
  const g = t.github;
  const [open, setOpen] = useState(false);
  const [status, setStatus] = useState<HubTokenStatus | null>(null);
  const [preview, setPreview] = useState<GitHubPreview | null>(null);
  const [token, setToken] = useState("");
  const [repo, setRepo] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [url, setUrl] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    let alive = true;
    setBusy(true); setError(null); setPreview(null); setStatus(null); setUrl(null);
    Promise.all([fetchGitHubToken(), fetchGitHubPreview(runId)])
      .then(([tokenStatus, snapshot]) => {
        if (alive) { setStatus(tokenStatus); setPreview(snapshot); }
      })
      .catch((e: unknown) => { if (alive) setError(reason(e, g.failed)); })
      .finally(() => { if (alive) setBusy(false); });
    return () => { alive = false; };
  }, [open, runId, g.failed]);

  const credentials = async (remove: boolean) => {
    setBusy(true); setError(null);
    const secret = token;
    setToken("");
    try {
      setStatus(remove ? await deleteGitHubToken() : await saveGitHubToken(secret));
    } catch (e) { setError(reason(e, g.failed)); }
    finally { setBusy(false); }
  };

  const upload = async () => {
    if (!preview || !status?.configured) return;
    setBusy(true); setError(null); setUrl(null);
    try {
      const result = await commitRunToGitHub(runId, repo.trim(), preview.digest);
      const destination = new URL(result.url);
      if (destination.protocol !== "https:" || !GITHUB_HOST.test(destination.hostname)) throw new Error();
      setUrl(destination.href);
    } catch (e) { setError(reason(e, g.failed)); }
    finally { setBusy(false); }
  };

  if (!open) return <button type="button" onClick={() => setOpen(true)}>{g.open}</button>;
  return <section aria-label={g.title} style={{ border: "1px solid var(--vf-panel-stroke)", borderRadius: 10, padding: 16 }}>
    <h3>{g.title}</h3>
    <button type="button" onClick={() => { setToken(""); setOpen(false); }} disabled={busy}>{g.close}</button>
    <p>{g.intro}</p>
    {busy && !preview && <p role="status">{g.loading}</p>}
    {status?.shared && <p role="alert">{g.shared}</p>}
    {status && <div>
      {status.configured && <p>{g.configured}</p>}
      <label>{g.token} <input type="password" autoComplete="new-password" value={token}
        onChange={(e) => setToken(e.target.value)} disabled={busy} style={FIELD} /></label>
      <p>{g.tokenHint}</p>
      <button type="button" disabled={busy || !token.trim()} onClick={() => void credentials(false)}>{g.saveToken}</button>
      {status.configured && <button type="button" disabled={busy} onClick={() => void credentials(true)}>{g.removeToken}</button>}
    </div>}
    {preview && <div>
      <h4>{g.preview}</h4>
      <p>{g.folder(preview.folder)}</p>
      <ul>{preview.files.map((file) => <li key={file.path}>{file.path} · {formatBytes(file.size, locale)}</li>)}</ul>
      <p>{g.noWeights}</p>
      <details><summary>{g.readme}</summary>
        <pre style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>{preview.readme}</pre>
      </details>
      <label>{g.repo} <input value={repo} onChange={(e) => { setRepo(e.target.value); setUrl(null); }}
        disabled={busy} placeholder={g.repoPlaceholder} style={FIELD} /></label>
      <p>{g.confirm}</p>
      <button type="button" disabled={busy || !status?.configured || !repo.trim()} onClick={() => void upload()}>
        {busy ? g.busy : g.send}
      </button>
    </div>}
    {error && <p role="alert">{error}</p>}
    {url && <a href={url} target="_blank" rel="noopener noreferrer">{g.success}</a>}
  </section>;
}

/** The server's translated reason when it sent one; else the generic failure. */
function reason(error: unknown, fallback: string): string {
  return error instanceof Error && error.message ? error.message : fallback;
}
