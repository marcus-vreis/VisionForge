import { useState } from "react";
import { updateProfile } from "../api/client";
import { useT } from "../i18n/useT";
import type { ProfileInfo } from "../lib/profile";
import { ROUTE, useRouteOpen } from "../lib/session";
import { CometProfileSettings } from "./CometProfileSettings";
import { TRACKER_HINT } from "./trackerStyles";
import { TrackerSwitch } from "./TrackerSwitch";
import { MlflowIcon } from "./ui/BrandIcons";
import { WandbProfileSettings } from "./WandbProfileSettings";

/** Edit one profile's exporters: local MLflow, Weights & Biases and Comet.
 *
 *  Lives in the header's profile menu: each tracker is a switch, and the ones
 *  that need a destination show their fields while on. Local MLflow is left
 *  out on a shared server, where nobody could open its store (ADR-223). */
export function ProfileTrackingSettings({ profile, onSaved }: {
  profile: ProfileInfo; onSaved: (profile: ProfileInfo) => void;
}) {
  const t = useT();
  const localMlflow = useRouteOpen(ROUTE.mlflowRetry);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(false);
  const change = async (enabled: boolean) => {
    setSaving(true); setError(false);
    try { onSaved(await updateProfile(profile.slug, { settings: { mlflow: { enabled } } })); }
    catch { setError(true); }
    finally { setSaving(false); }
  };
  return <div style={{ display: "flex", flexDirection: "column", gap: 16, fontSize: 13, color: "var(--vf-text-dim)" }}>
    {localMlflow && <div data-mlflow-switch style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      <TrackerSwitch icon={<MlflowIcon size={18} />} label={t.mlflow.label}
        checked={profile.settings?.mlflow.enabled ?? false} disabled={saving}
        onChange={(enabled) => void change(enabled)} />
      <div style={TRACKER_HINT}>{t.mlflow.hint}</div>
      {error && <div role="alert">{t.mlflow.saveFailed}</div>}
    </div>}
    <WandbProfileSettings profile={profile} onSaved={onSaved} />
    <CometProfileSettings profile={profile} onSaved={onSaved} />
  </div>;
}
