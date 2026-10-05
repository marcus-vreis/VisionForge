import type { Dict } from "../i18n/pt";
import type { DatasetInfo } from "../types/run";

export type DatasetVerdict =
  | { kind: "same" }
  | { kind: "different" }
  | { kind: "unknown"; reason: string };

/** Whether two runs saw the same data — mirrors `same_dataset` in Python.
 *
 * Deliberately duplicated rather than served by an endpoint: the rule is four
 * lines, and comparing two fields over HTTP would cost more than the copy. What
 * must survive the translation is the third answer — most runs predate the
 * fingerprint, so "cannot tell" is the common case rather than an edge one, and
 * reporting it as "different" would be worse than saying nothing.
 */
export function compareDatasets(
  t: Dict,
  a: DatasetInfo | null | undefined,
  b: DatasetInfo | null | undefined,
): DatasetVerdict {
  if (!a?.digest || !b?.digest) {
    return {
      kind: "unknown",
      reason: t.datasetIdentity.noFingerprint,
    };
  }
  if (a.method !== b.method) {
    return { kind: "unknown", reason: t.datasetIdentity.differentMethod };
  }
  return a.digest === b.digest ? { kind: "same" } : { kind: "different" };
}

const UNITS = ["B", "KB", "MB", "GB", "TB"];

/** `123456789` → `117,7 MB` in pt-BR and `117.7 MB` in en-US: the decimal
 *  separator follows `locale`, the one `useI18n()` hands out. */
export function formatBytes(
  bytes: number | null | undefined,
  locale: string,
): string {
  if (bytes == null) return "—";
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < UNITS.length - 1) {
    value /= 1024;
    unit += 1;
  }
  const shown =
    unit === 0
      ? String(value)
      : new Intl.NumberFormat(locale, {
          minimumFractionDigits: 1,
          maximumFractionDigits: 1,
          useGrouping: false,
        }).format(value);
  return `${shown} ${UNITS[unit]}`;
}

/** The first 12 characters — enough to tell two digests apart by eye. */
export function shortDigest(digest: string | null | undefined): string {
  return digest ? digest.slice(0, 12) : "—";
}
