import { useState, type CSSProperties } from "react";
import {
  fetchDvcPreview, prepareDvcRepository, versionRunWithDvc,
  type DvcExportResult, type DvcPreview,
} from "../api/client";
import { formatBytes } from "../lib/dataset-identity";
import { canPrepare, repositoryNotes, resultLines } from "../lib/dvc-export";
import { useI18n } from "../i18n/useT";

// The last repository folder, remembered in this browser only (a convenience).
const REPO_KEY = "vf.dvc.repo";
const FIELD: CSSProperties = {
  background: "var(--vf-panel)", color: "var(--vf-text)", padding: 8,
  border: "1px solid var(--vf-panel-stroke)", borderRadius: 6, minWidth: 0, width: "100%",
};

/** Version a run with DVC in a local repository of experiments (ADR-176). */
export function DvcExportPanel({ runId }: { runId: string }) {
  const { t, locale } = useI18n();
  const d = t.dvc;
  const [open, setOpen] = useState(false);
  const [repo, setRepo] = useState(readRepo);
  const [preview, setPreview] = useState<DvcPreview | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<DvcExportResult | null>(null);

  const run = async (step: () => Promise<void>) => {
    setBusy(true); setError(null);
    try { await step(); } catch (e) { setError(reason(e, d.failed)); } finally { setBusy(false); }
  };
  const look = () => run(async () => {
    setResult(null);
    const folder = repo.trim();
    rememberRepo(folder);
    setPreview(await fetchDvcPreview(runId, folder));
  });
  const prepare = () => run(async () => {
    const folder = repo.trim();
    await prepareDvcRepository(folder);
    setPreview(await fetchDvcPreview(runId, folder));
  });
  const send = () => run(async () => {
    if (!preview) return;
    setResult(await versionRunWithDvc(runId, repo.trim(), preview.digest));
  });

  if (!open) return <button type="button" onClick={() => setOpen(true)}>{d.open}</button>;
  const repository = preview?.repository;
  const ready = repository?.state === "ready";
  return <section aria-label={d.title} style={{ border: "1px solid var(--vf-panel-stroke)", borderRadius: 10, padding: 16 }}>
    <h3>{d.title}</h3>
    <button type="button" onClick={() => setOpen(false)} disabled={busy}>{d.close}</button>
    <p>{d.intro}</p>
    <label>{d.repo} <input value={repo} placeholder={d.repoPlaceholder} disabled={busy} style={FIELD}
      onChange={(e) => { setRepo(e.target.value); setPreview(null); setResult(null); }} /></label>
    <button type="button" disabled={busy || !repo.trim()} onClick={() => void look()}>{d.check}</button>
    {busy && !preview && <p role="status">{d.loading}</p>}
    {repository && repositoryNotes(repository, d).map((note) => <p key={note}>{note}</p>)}
    {repository && canPrepare(repository) && <button type="button" disabled={busy} onClick={() => void prepare()}>
      {d.prepare}
    </button>}
    {preview && <div>
      <h4>{d.preview}</h4>
      <p>{d.folder(preview.folder)}</p>
      <ul>{preview.files.map((file) => <li key={file.path}>
        {file.path} · {formatBytes(file.size, locale)} · {file.storage === "dvc" ? d.storageDvc : d.storageGit}
      </li>)}</ul>
      <details><summary>{d.readme}</summary>
        <pre style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>{preview.readme}</pre>
      </details>
      {ready && <>
        <p>{d.confirm}</p>
        <button type="button" disabled={busy} onClick={() => void send()}>{busy ? d.busy : d.send}</button>
      </>}
    </div>}
    {error && <p role="alert">{error}</p>}
    {result && <div role="status">{resultLines(result, d).map((line) => <p key={line}>{line}</p>)}</div>}
  </section>;
}

function readRepo(): string {
  try { return localStorage.getItem(REPO_KEY) ?? ""; } catch { return ""; }
}

function rememberRepo(folder: string): void {
  try { localStorage.setItem(REPO_KEY, folder); } catch { /* a convenience only */ }
}

/** The server's translated reason when it sent one; else the generic failure. */
function reason(error: unknown, fallback: string): string {
  return error instanceof Error && error.message ? error.message : fallback;
}
