/**
 * Pure helpers for the grid-search per-hyperparameter axis editor.
 *
 * A grid axis for a field is the list of candidate values swept by
 * `GridSearchBlock`: `grid_search.hyperparameters["training.learning_rate"] =
 * [0.001, 0.0005]`. These functions decide which fields can become an axis,
 * coerce raw text entries, validate each candidate against the field's schema,
 * and suggest the next value when the user clicks "+".
 */
import type { ControlKind } from "../components/field-renderer";
import type { Dict } from "../i18n/pt";
import type { JsonSchema } from "../types/schema";

/** Field names never offered as a grid axis even though their control qualifies. */
export const GRID_EXCLUDED_FIELDS = new Set(["task", "num_classes", "seed"]);

/** A field can become a grid axis only if it is a scalar number/enum control. */
export function isGridableField(name: string, kind: ControlKind): boolean {
  if (GRID_EXCLUDED_FIELDS.has(name)) return false;
  return kind === "number" || kind === "select" || kind === "segmented";
}

/** Coerce a raw text entry to a number when it parses, else keep the string so
 * inline validation can flag it. Enum values stay as strings. */
export function coerceGridValue(raw: string, isEnum: boolean): unknown {
  if (isEnum) return raw;
  const trimmed = raw.trim();
  const n = Number(trimmed);
  return trimmed === "" || Number.isNaN(n) ? raw : n;
}

/** Validate one candidate grid value against the field schema + known rules.
 * Returns an inline error message in the language of `t`, or null when valid. */
export function validateGridValue(
  t: Dict,
  name: string,
  schema: JsonSchema,
  value: unknown,
): string | null {
  const msg = t.gridAxis;
  if (schema.enum) {
    return schema.enum.map(String).includes(String(value))
      ? null
      : msg.notAnOption;
  }
  const n = typeof value === "number" ? value : Number(value);
  if (value === "" || value === null || Number.isNaN(n)) return msg.invalidValue;
  if (schema.type === "integer" && !Number.isInteger(n))
    return msg.mustBeInteger;
  if (typeof schema.exclusiveMinimum === "number" && n <= schema.exclusiveMinimum)
    return msg.mustBeAbove(schema.exclusiveMinimum);
  if (typeof schema.minimum === "number" && n < schema.minimum)
    return msg.mustBeAtLeast(schema.minimum);
  if (name === "batch_size" && (n < 1 || (n & (n - 1)) !== 0))
    return msg.mustBePowerOfTwo;
  return null;
}

const MODEL_AXIS = "model.name";
const RATE_AXIS = "training.learning_rate";
/** The factors a new multiplier axis starts with (ADR-158). */
export const DEFAULT_LR_MULTIPLIERS = [0.3, 1, 3];

type GridSection = Record<string, unknown>;

const axesOf = (grid: GridSection) =>
  (grid["hyperparameters"] ?? {}) as Record<string, unknown[]>;

/** Whether the rate axis can be read as a factor: there are models to scale. */
export function canUseLrMultiplier(hyperparameters: Record<string, unknown[]>): boolean {
  return (hyperparameters[MODEL_AXIS]?.length ?? 0) > 0;
}

/** Set or drop one axis. A multiplier that loses either axis ends with the
 *  factors removed too: left behind, 3 would be read as a learning rate of 3. */
export function withGridAxis(
  grid: GridSection,
  dotPath: string,
  values: unknown[] | null,
): GridSection {
  const axes = { ...axesOf(grid) };
  if (values === null) delete axes[dotPath];
  else axes[dotPath] = values;
  if (grid["lr_mode"] !== "suggested_multiplier" || (axes[MODEL_AXIS] && axes[RATE_AXIS])) {
    return { ...grid, hyperparameters: axes };
  }
  delete axes[RATE_AXIS];
  return { ...grid, hyperparameters: axes, lr_mode: "absolute" };
}

/** Switch how the rate axis is read. Its values mean different things in each
 *  mode, so they are never carried across. */
export function withLrMultiplier(grid: GridSection, on: boolean): GridSection {
  const axes = { ...axesOf(grid) };
  if (on) axes[RATE_AXIS] = [...DEFAULT_LR_MULTIPLIERS];
  else delete axes[RATE_AXIS];
  return {
    ...grid,
    hyperparameters: axes,
    lr_mode: on ? "suggested_multiplier" : "absolute",
  };
}

/** Suggest the next candidate value when the user clicks "+ valor". */
export function suggestNextGridValue(
  prev: unknown,
  name: string,
  schema: JsonSchema,
): unknown {
  if (schema.enum && schema.enum.length) {
    const alt = schema.enum.find((e) => String(e) !== String(prev));
    return alt ?? schema.enum[0];
  }
  if (name === "batch_size" && typeof prev === "number") return prev * 2;
  if (typeof prev === "number")
    return schema.type === "integer" ? prev + 1 : prev / 2;
  return prev;
}
