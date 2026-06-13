/** Pure helpers that turn the SweepCard editor rows into the backend
 *  `search_space` payload (ADR-045). Kept out of the component so the parsing
 *  and the grid-trial-count preview are unit-testable. */

export type SweepMode = "grid" | "random";
export type RandomKind = "uniform" | "log_uniform" | "choice";

export interface SweepRow {
  path: string;
  /** grid: comma-separated value list. */
  values: string;
  /** random: distribution kind + its params. */
  kind: RandomKind;
  low: string;
  high: string;
  /** random choice: comma-separated options. */
  options: string;
}

export function makeSweepRow(): SweepRow {
  return { path: "", values: "", kind: "uniform", low: "", high: "", options: "" };
}

/** Coerce a token to a number when it parses cleanly, else keep the string. */
function coerce(token: string): number | string {
  const t = token.trim();
  const n = Number(t);
  return t !== "" && !Number.isNaN(n) ? n : t;
}

function parseList(s: string): (number | string)[] {
  return s
    .split(",")
    .map(coerce)
    .filter((v) => v !== "");
}

/** Build the backend `search_space` from editor rows (empty paths skipped). */
export function buildSearchSpace(
  mode: SweepMode,
  rows: SweepRow[],
): Record<string, unknown> {
  const space: Record<string, unknown> = {};
  for (const row of rows) {
    const path = row.path.trim();
    if (!path) continue;
    if (mode === "grid") {
      const values = parseList(row.values);
      if (values.length > 0) space[path] = values;
    } else if (row.kind === "choice") {
      const options = parseList(row.options);
      if (options.length > 0) space[path] = { type: "choice", options };
    } else {
      const low = Number(row.low);
      const high = Number(row.high);
      if (row.low.trim() !== "" && row.high.trim() !== "" && !Number.isNaN(low) && !Number.isNaN(high)) {
        space[path] = { type: row.kind, low, high };
      }
    }
  }
  return space;
}

/** Cartesian-product trial count for the grid preview banner (0 = nothing set). */
export function gridTrialCount(rows: SweepRow[]): number {
  let total = 1;
  let any = false;
  for (const row of rows) {
    if (!row.path.trim()) continue;
    const count = parseList(row.values).length;
    if (count > 0) {
      total *= count;
      any = true;
    }
  }
  return any ? total : 0;
}
