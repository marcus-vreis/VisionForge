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
  a: DatasetInfo | null | undefined,
  b: DatasetInfo | null | undefined,
): DatasetVerdict {
  if (!a?.digest || !b?.digest) {
    return {
      kind: "unknown",
      reason: "um dos runs não tem fingerprint (anterior a 26/07/2026)",
    };
  }
  if (a.method !== b.method) {
    return { kind: "unknown", reason: "os dois runs usaram método diferente" };
  }
  return a.digest === b.digest ? { kind: "same" } : { kind: "different" };
}

const UNITS = ["B", "KB", "MB", "GB", "TB"];

/** `123456789` → `117,7 MB`. Comma is the decimal separator in pt-BR. */
export function formatBytes(bytes: number | null | undefined): string {
  if (bytes == null) return "—";
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < UNITS.length - 1) {
    value /= 1024;
    unit += 1;
  }
  const shown = unit === 0 ? String(value) : value.toFixed(1).replace(".", ",");
  return `${shown} ${UNITS[unit]}`;
}

/** The first 12 characters — enough to tell two digests apart by eye. */
export function shortDigest(digest: string | null | undefined): string {
  return digest ? digest.slice(0, 12) : "—";
}
