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
 * Returns an inline error message, or null when valid. */
export function validateGridValue(
  name: string,
  schema: JsonSchema,
  value: unknown,
): string | null {
  if (schema.enum) {
    return schema.enum.map(String).includes(String(value))
      ? null
      : "fora das opções";
  }
  const n = typeof value === "number" ? value : Number(value);
  if (value === "" || value === null || Number.isNaN(n)) return "valor inválido";
  if (schema.type === "integer" && !Number.isInteger(n))
    return "precisa ser inteiro";
  if (typeof schema.exclusiveMinimum === "number" && n <= schema.exclusiveMinimum)
    return `precisa ser > ${schema.exclusiveMinimum}`;
  if (typeof schema.minimum === "number" && n < schema.minimum)
    return `precisa ser ≥ ${schema.minimum}`;
  if (name === "batch_size" && (n < 1 || (n & (n - 1)) !== 0))
    return "precisa ser potência de 2";
  return null;
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
