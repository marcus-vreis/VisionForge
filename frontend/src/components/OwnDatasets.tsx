import { useCallback, useEffect, useState, type CSSProperties, type ReactNode } from "react";

import { ApiError, deleteOwnDataset, fetchOwnDatasets, fetchSession, type OwnDataset } from "../api/client";
import { useI18n } from "../i18n/useT";
import {
  ROUTE,
  diskSize,
  gigabytes,
  quotaFull,
  routeOpen,
  setSession,
  useSession,
} from "../lib/session";
import { OwnDatasetUpload } from "./OwnDatasetUpload";

type Load = "loading" | "ready" | "failed";

const BUTTON: CSSProperties = {
  padding: "7px 14px",
  borderRadius: 8,
  border: "1px solid var(--vf-panel-stroke)",
  background: "transparent",
  color: "var(--vf-text)",
  fontFamily: "var(--font-mono)",
  fontSize: 11,
  letterSpacing: "0.10em",
  textTransform: "uppercase",
  cursor: "pointer",
};

/** The datasets in the person's own folder on a shared server (ADR-227).
 *
 * What they downloaded or derived counts against their disk quota: here they
 * see each one's size and delete what they no longer need. Deleting asks
 * twice, in the row itself, since it cannot be undone. A dataset from their
 * own computer arrives here as a .zip (ADR-246).
 */
export function OwnDatasets() {
  const { t } = useI18n();
  const canUpload = routeOpen(useSession(), ROUTE.datasetUpload);
  const [load, setLoad] = useState<Load>("loading");
  const [datasets, setDatasets] = useState<OwnDataset[]>([]);
  const [asking, setAsking] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setDatasets(await fetchOwnDatasets());
      setLoad("ready");
    } catch {
      setLoad("failed");
    }
  }, []);

  useEffect(() => {
    let alive = true;
    fetchOwnDatasets()
      .then((list) => {
        if (!alive) return;
        setDatasets(list);
        setLoad("ready");
      })
      .catch(() => {
        if (alive) setLoad("failed");
      });
    return () => {
      alive = false;
    };
  }, []);

  const remove = async (name: string) => {
    setDeleting(name);
    setError(null);
    setNotice(null);
    try {
      await deleteOwnDataset(name);
      setNotice(t.ownDatasets.deleted(name));
      await refresh();
      // The profile menu's disk figure comes from the session: read it again.
      setSession(await fetchSession());
    } catch (e) {
      setError(e instanceof ApiError ? e.message : t.ownDatasets.failed);
    } finally {
      setDeleting(null);
      setAsking(null);
    }
  };

  const afterUpload = async () => {
    await refresh();
    try {
      // The new dataset counts in the profile menu's disk figure.
      setSession(await fetchSession());
    } catch {
      // The list is right; the figure catches up on the next load.
    }
  };

  return (
    <OwnDatasetsView
      upload={canUpload ? <OwnDatasetUpload onUploaded={afterUpload} /> : null}
      load={load}
      datasets={datasets}
      asking={asking}
      deleting={deleting}
      error={error}
      notice={notice}
      onAsk={(name) => {
        setError(null);
        setNotice(null);
        setAsking(name);
      }}
      onDelete={(name) => void remove(name)}
    />
  );
}

interface OwnDatasetsViewProps {
  load: Load;
  datasets: OwnDataset[];
  /** The dataset whose row is asking "delete for good?", if any. */
  asking: string | null;
  /** The one being deleted now; every button waits for it. */
  deleting: string | null;
  error: string | null;
  notice: string | null;
  onAsk: (name: string | null) => void;
  onDelete: (name: string) => void;
  /** "Send .zip", where the server takes uploads. */
  upload?: ReactNode;
}

/** What the list shows for a given state; the tests render it without a server. */
export function OwnDatasetsView({
  load,
  datasets,
  asking,
  deleting,
  error,
  notice,
  onAsk,
  onDelete,
  upload = null,
}: OwnDatasetsViewProps) {
  const { t, locale } = useI18n();
  const quota = useSession().quota;
  return (
    <section
      data-own-datasets
      aria-label={t.ownDatasets.title}
      style={{ display: "flex", flexDirection: "column", gap: 14, fontSize: 14 }}
    >
      <p style={{ margin: 0, lineHeight: 1.55, color: "var(--vf-text-dim)" }}>
        {t.ownDatasets.intro}
      </p>
      {quota && (
        <div
          data-quota={quotaFull(quota) ? "full" : "ok"}
          style={{ fontFamily: "var(--font-mono)", fontSize: 12, color: "var(--vf-text)" }}
        >
          {t.server.quota(
            gigabytes(quota.used_bytes, locale),
            gigabytes(quota.limit_bytes, locale),
          )}
        </div>
      )}
      {upload}
      {notice && <div role="status">{notice}</div>}
      {error && <div role="alert">{error}</div>}
      {load === "loading" && <div>{t.ownDatasets.loading}</div>}
      {load === "failed" && <div role="alert">{t.ownDatasets.loadFailed}</div>}
      {load === "ready" && datasets.length === 0 && (
        <div style={{ color: "var(--vf-text-dim)" }}>{t.ownDatasets.empty}</div>
      )}
      {load === "ready" && datasets.length > 0 && (
        <ul style={{ listStyle: "none", margin: 0, padding: 0, display: "flex", flexDirection: "column", gap: 8 }}>
          {datasets.map((d) => (
            <li
              key={d.name}
              style={{
                display: "flex",
                alignItems: "center",
                flexWrap: "wrap",
                gap: 12,
                padding: "10px 12px",
                border: "1px solid var(--vf-panel-stroke)",
                borderRadius: 10,
              }}
            >
              <div style={{ flex: "1 1 220px", minWidth: 0 }}>
                <div style={{ color: "var(--vf-text)", overflowWrap: "anywhere" }}>{d.name}</div>
                <div
                  style={{
                    fontFamily: "var(--font-mono)",
                    fontSize: 12,
                    color: "var(--vf-text-dim)",
                    overflowWrap: "anywhere",
                  }}
                >
                  {diskSize(d.size_bytes, locale)} · {d.path}
                </div>
              </div>
              {asking === d.name ? (
                <div style={{ display: "flex", alignItems: "center", flexWrap: "wrap", gap: 8 }}>
                  <span>{t.ownDatasets.confirm}</span>
                  <button
                    type="button"
                    style={BUTTON}
                    disabled={deleting !== null}
                    onClick={() => onDelete(d.name)}
                  >
                    {deleting === d.name ? t.ownDatasets.deleting : t.ownDatasets.confirmYes}
                  </button>
                  <button
                    type="button"
                    style={BUTTON}
                    disabled={deleting !== null}
                    onClick={() => onAsk(null)}
                  >
                    {t.ownDatasets.confirmNo}
                  </button>
                </div>
              ) : (
                <button
                  type="button"
                  style={BUTTON}
                  aria-label={t.ownDatasets.deleteLabel(d.name)}
                  disabled={deleting !== null}
                  onClick={() => onAsk(d.name)}
                >
                  {t.ownDatasets.delete}
                </button>
              )}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
