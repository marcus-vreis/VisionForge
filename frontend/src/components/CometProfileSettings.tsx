import { useEffect, useState, type CSSProperties } from "react";
import {
  deleteCometToken, fetchCometToken, saveCometToken, updateProfile,
  type CometSettings, type HubTokenStatus,
} from "../api/client";
import { useT } from "../i18n/useT";
import type { ProfileInfo } from "../lib/profile";

const FIELD: CSSProperties = {
  background: "var(--vf-panel)", color: "var(--vf-text)", padding: 6,
  border: "1px solid var(--vf-panel-stroke)", borderRadius: 6, maxWidth: "100%",
};
const DEFAULTS: CometSettings = { enabled: false, project: "visionforge", workspace: "" };

/** One profile's Comet export: on/off, private destination and API key (ADR-200). */
export function CometProfileSettings({ profile, onSaved }: {
  profile: ProfileInfo; onSaved: (profile: ProfileInfo) => void;
}) {
  const t = useT();
  const saved = profile.settings?.comet ?? DEFAULTS;
  const [project, setProject] = useState(saved.project);
  const [workspace, setWorkspace] = useState(saved.workspace);
  const [token, setToken] = useState("");
  const [keyStatus, setKeyStatus] = useState<HubTokenStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!saved.enabled) return;
    let alive = true;
    fetchCometToken(profile.slug)
      .then((status) => { if (alive) setKeyStatus(status); })
      .catch(() => { if (alive) setKeyStatus(null); });
    return () => { alive = false; };
  }, [profile.slug, saved.enabled]);

  const save = async (change: Partial<CometSettings>) => {
    setBusy(true); setError(null);
    try { onSaved(await updateProfile(profile.slug, { settings: { comet: change } })); }
    catch (e) { setError(e instanceof Error && e.message ? e.message : t.comet.saveFailed); }
    finally { setBusy(false); }
  };

  const credentials = async (remove: boolean) => {
    setBusy(true); setError(null);
    const secret = token;
    setToken("");
    try {
      setKeyStatus(remove
        ? await deleteCometToken(profile.slug)
        : await saveCometToken(profile.slug, secret));
    } catch (e) { setError(e instanceof Error && e.message ? e.message : t.comet.tokenFailed); }
    finally { setBusy(false); }
  };

  const changed = project.trim() !== saved.project || workspace.trim() !== saved.workspace;
  return <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
    <label><input type="checkbox" checked={saved.enabled} disabled={busy}
      onChange={(e) => void save({ enabled: e.target.checked })} /> {t.comet.label}</label>
    <div>{t.comet.hint}</div>
    {saved.enabled && <>
      <label>{t.comet.project} <input value={project} disabled={busy} style={FIELD}
        onChange={(e) => setProject(e.target.value)} /></label>
      <label>{t.comet.workspace} <input value={workspace} disabled={busy} style={FIELD}
        placeholder={t.comet.workspaceHint} onChange={(e) => setWorkspace(e.target.value)} /></label>
      <button type="button" disabled={busy || !changed || !project.trim()}
        onClick={() => void save({ project: project.trim(), workspace: workspace.trim() })}>{t.comet.saveTarget}</button>
      {keyStatus?.shared && <div role="alert">{t.comet.shared}</div>}
      {keyStatus?.configured && <div>{t.comet.configured}</div>}
      <label>{t.comet.token} <input type="password" autoComplete="new-password" value={token}
        disabled={busy} style={FIELD} onChange={(e) => setToken(e.target.value)} /></label>
      <div>{t.comet.tokenHint}</div>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        <button type="button" disabled={busy || !token.trim()}
          onClick={() => void credentials(false)}>{t.comet.saveToken}</button>
        {keyStatus?.configured && <button type="button" disabled={busy}
          onClick={() => void credentials(true)}>{t.comet.removeToken}</button>}
      </div>
    </>}
    {error && <div role="alert">{error}</div>}
  </div>;
}
