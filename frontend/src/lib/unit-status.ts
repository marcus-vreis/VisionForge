/** How a unit of a multi-unit job (a fold, a model, a replicate, a trial) ended.
 *
 * The server records `success`, `failed`, or — since ADR-111 — `stopped`: the
 * unit that was running when the researcher asked to stop. It keeps whatever
 * metrics it reached and is left out of every aggregate and ranking, which is
 * why it must not read as a failure (nothing went wrong, and there is no error
 * to show), nor as a success (its numbers belong to a configuration that never
 * finished).
 */
export type UnitState = "ok" | "stopped" | "failed";

export function unitState(status: string | null | undefined): UnitState {
  if (status === "success") return "ok";
  if (status === "stopped") return "stopped";
  return "failed";
}

/** The amber of a stopped unit: apart from the red of a failure and the green of
 *  a success. */
export const STOPPED_COLOR = "oklch(0.84 0.12 85)";

/** How many units in `rows` ended in `state`. */
export function countUnits(
  rows: ReadonlyArray<{ status: string }>,
  state: UnitState,
): number {
  return rows.filter((row) => unitState(row.status) === state).length;
}
