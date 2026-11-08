import { useState } from "react";
import { updateProfile } from "../api/client";
import { useT } from "../i18n/useT";
import type { ProfileInfo } from "../lib/profile";

/** Edit one profile's local exporter preference. */
export function ProfileTrackingSettings({ profile, onSaved }: {
  profile: ProfileInfo; onSaved: (profile: ProfileInfo) => void;
}) {
  const t = useT();
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(false);
  const change = async (enabled: boolean) => {
    setSaving(true); setError(false);
    try { onSaved(await updateProfile(profile.slug, { settings: { mlflow: { enabled } } })); }
    catch { setError(true); }
    finally { setSaving(false); }
  };
  return <div style={{ fontSize: 12, color: "var(--vf-text-dim)", padding: 8 }}>
    <label><input type="checkbox" checked={profile.settings?.mlflow.enabled ?? false}
      disabled={saving} onChange={(e) => void change(e.target.checked)} /> {t.mlflow.label}</label>
    <div>{t.mlflow.hint}</div>
    {error && <div role="alert">{t.mlflow.saveFailed}</div>}
  </div>;
}
