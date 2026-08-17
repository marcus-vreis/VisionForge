/**
 * Schema walking for the generic custom-task form (ADR-058).
 *
 * These live outside `SchemaForm.tsx` because they are not components: a module
 * that exports both loses Fast Refresh, so editing the form during development
 * would remount it and throw away whatever the researcher had typed.
 */

import { resolveKind } from "../components/field-renderer";
import type { JsonSchema } from "../types/schema";

/** Canonical order of the blocks every custom task inherits from BaseTaskConfig. */
export const SECTION_ORDER = [
  "model",
  "training",
  "data",
  "transforms",
  "preprocessing",
  "output",
];

/** Follow $ref / anyOf until a concrete schema is reached. */
export function resolveSchema(
  schema: JsonSchema,
  defs: Record<string, JsonSchema>,
): JsonSchema {
  if (schema.$ref) {
    const name = schema.$ref.split("/").pop();
    const target = name ? defs[name] : undefined;
    return target ? resolveSchema(target, defs) : schema;
  }
  if (schema.anyOf) {
    const nonNull = schema.anyOf.find((s) => s.type !== "null");
    if (nonNull) return resolveSchema(nonNull, defs);
  }
  return schema;
}

/** Properties of an object schema that will actually render a control. */
export function visibleChildren(
  schema: JsonSchema,
  defs: Record<string, JsonSchema>,
): [string, JsonSchema][] {
  if (schema.type !== "object" || !schema.properties) return [];
  const out: [string, JsonSchema][] = [];
  for (const [name, child] of Object.entries(schema.properties)) {
    const resolved = resolveSchema(child, defs);
    const kind = resolveKind(name, resolved);
    if (kind === "skip") continue;
    // Recurse: a block whose children are all skipped is itself invisible.
    if (kind === "object" && visibleChildren(resolved, defs).length === 0) continue;
    out.push([name, child]);
  }
  return out;
}

/** Canonical order first, then whatever else the schema declared. */
export function orderSections(names: string[]): string[] {
  const known = SECTION_ORDER.filter((n) => names.includes(n));
  const rest = names.filter((n) => !SECTION_ORDER.includes(n));
  return [...known, ...rest];
}
