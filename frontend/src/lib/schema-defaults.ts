import type { JsonSchema } from "../types/schema";

/** Build default values from a JSON Schema recursively. */
export function buildDefaults(
  schema: JsonSchema,
  defs: Record<string, JsonSchema>,
): unknown {
  if (schema.$ref) {
    const refName = schema.$ref.split("/").pop()!;
    return buildDefaults(defs[refName], defs);
  }
  if (schema.default !== undefined) return schema.default;
  if (schema.anyOf) {
    const nonNull = schema.anyOf.find((s) => s.type !== "null");
    if (nonNull) return buildDefaults(nonNull, defs);
    return null;
  }
  if (schema.type === "object" && schema.properties) {
    const obj: Record<string, unknown> = {};
    for (const [key, prop] of Object.entries(schema.properties)) {
      obj[key] = buildDefaults(prop, defs);
    }
    return obj;
  }
  if (schema.enum) return schema.enum[0];
  if (schema.type === "string") return "";
  if (schema.type === "integer" || schema.type === "number") return 0;
  if (schema.type === "boolean") return false;
  if (schema.type === "array") return [];
  return null;
}
