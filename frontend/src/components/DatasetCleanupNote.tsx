import type { CSSProperties } from "react";

import { useI18n } from "../i18n/useT";
import { diskSize } from "../lib/session";
import type { DatasetCleanup } from "../types/run";

/** One line per dataset the server deleted once the run ended (ADR-249).
 *
 * On a shared server a dataset sent to go after its next run is deleted when
 * that run ends; the run page and the run's History entry say so. Nothing is
 * shown when the run deleted nothing.
 */
export function DatasetCleanupNote({
  entries,
  style,
}: {
  entries?: DatasetCleanup[] | null;
  /** Merged into the block's own style: where it sits on the page. */
  style?: CSSProperties;
}) {
  const { t, locale } = useI18n();
  if (!entries || entries.length === 0) return null;
  return (
    <div
      data-dataset-cleanup
      role="status"
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 4,
        fontFamily: "var(--font-mono)",
        fontSize: 12,
        lineHeight: 1.55,
        color: "var(--vf-text-dim)",
        overflowWrap: "anywhere",
        ...style,
      }}
    >
      {entries.map((entry) => (
        <div key={`${entry.path}:${entry.at}`} data-cleanup-status={entry.status}>
          {entry.status === "deleted"
            ? t.datasetCleanup.deleted(entry.dataset, diskSize(entry.size_bytes, locale))
            : t.datasetCleanup.failed(entry.dataset, entry.error ?? "")}
        </div>
      ))}
    </div>
  );
}
