import { useEffect, useState, type CSSProperties } from "react";
import {
  deleteWandbToken, fetchWandbToken, saveWandbToken, updateProfile,
  type HubTokenStatus, type WandbSettings,
} from "../api/client";
import { useT } from "../i18n/useT";
import type { ProfileInfo } from "../lib/profile";

const FIELD: CSSProperties = {
  background: "var(--vf-panel)", color: "var(--vf-text)", padding: 6,
  border: "1px solid var(--vf-panel-stroke)", borderRadius: 6, maxWidth: "100%",
};
const DEFAULTS: WandbSettings = { enabled: false, project: "visionforge", entity: "" };

/** One profile's W&B export: on/off, private destination and API key (ADR-172). */
export function WandbProfileSettings({ profile, onSaved }: {
  profile: ProfileInfo; onSaved: (profile: ProfileInfo) => void;
}) {
  const t = useT();
  const saved = profile.settings?.wandb ?? DEFAULTS;
  const [project, setProject] = useState(saved.project);
  const [entity, setEntity] = useState(saved.entity);
  const [token, setToken] = useState("");
  const [keyStatus, setKeyStatus] = useState<HubTokenStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!saved.enabled) return;
    let alive = true;
    fetchWandbToken(profile.slug)
      .then((status) => { if (alive) setKeyStatus(status); })
      .catch(() => { if (alive) setKeyStatus(null); });
    return () => { alive = false; };
  }, [profile.slug, saved.enabled]);

  const save = async (change: Partial<WandbSettings>) => {
    setBusy(true); setError(null);
    try { onSaved(await updateProfile(profile.slug, { settings: { wandb: change } })); }
    catch (e) { setError(e instanceof Error && e.message ? e.message : t.wandb.saveFailed); }
    finally { setBusy(false); }
  };

  const credentials = async (remove: boolean) => {
    setBusy(true); setError(null);
    const secret = token;
    setToken("");
    try {
      setKeyStatus(remove
        ? await deleteWandbToken(profile.slug)
        : await saveWandbToken(profile.slug, secret));
    } catch (e) { setError(e instanceof Error && e.message ? e.message : t.wandb.tokenFailed); }
    finally { setBusy(false); }
  };

  const changed = project.trim() !== saved.project || entity.trim() !== saved.entity;
  return <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
    <label><input type="checkbox" checked={saved.enabled} disabled={busy}
      onChange={(e) => void save({ enabled: e.target.checked })} /> {t.wandb.label}</label>
    <div>{t.wandb.hint}</div>
    {saved.enabled && <>
      <label>{t.wandb.project} <input value={project} disabled={busy} style={FIELD}
        onChange={(e) => setProject(e.target.value)} /></label>
      <label>{t.wandb.entity} <input value={entity} disabled={busy} style={FIELD}
        placeholder={t.wandb.entityHint} onChange={(e) => setEntity(e.target.value)} /></label>
      <button type="button" disabled={busy || !changed || !project.trim()}
        onClick={() => void save({ project: project.trim(), entity: entity.trim() })}>{t.wandb.saveTarget}</button>
      {keyStatus?.shared && <div role="alert">{t.wandb.shared}</div>}
      {keyStatus?.configured && <div>{t.wandb.configured}</div>}
      <label>{t.wandb.token} <input type="password" autoComplete="new-password" value={token}
        disabled={busy} style={FIELD} onChange={(e) => setToken(e.target.value)} /></label>
      <div>{t.wandb.tokenHint}</div>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        <button type="button" disabled={busy || !token.trim()}
          onClick={() => void credentials(false)}>{t.wandb.saveToken}</button>
        {keyStatus?.configured && <button type="button" disabled={busy}
          onClick={() => void credentials(true)}>{t.wandb.removeToken}</button>}
      </div>
    </>}
    {error && <div role="alert">{error}</div>}
  </div>;
}
