/**
 * Determines which custom control to use for a given JSON Schema field.
 * Returns a descriptor the form layer uses to pick the right component.
 */
import type { JsonSchema } from "../types/schema";

export type ControlKind =
  | "number"
  | "select"
  | "segmented"
  | "toggle"
  | "text"
  | "object"
  | "array-number"
  | "skip";

/** Small enum: optimizer fields always get Segmented; pretrained gets Toggle, etc. */
const SEGMENTED_FIELDS = new Set(["optimizer"]);
const SKIP_FIELDS = new Set([
  "models_dir",
  "graphics_dir",
  "logs_dir",
  "reports_dir",
  "checkpoint_path",
  "weights_path",
  "mode",
  // Device is owned by the Header/BottomBar DeviceSelector — don't render it
  // a second time in the form panel.
  "device",
]);

export function resolveKind(
  fieldName: string,
  schema: JsonSchema,
): ControlKind {
  if (SKIP_FIELDS.has(fieldName)) return "skip";

  if (schema.$ref) return "object";

  // anyOf — check inner non-null type
  if (schema.anyOf) {
    const nonNull = schema.anyOf.find((s) => s.type !== "null");
    if (!nonNull) return "skip";
    return resolveKind(fieldName, nonNull);
  }

  if (schema.type === "object") return "object";

  if (schema.enum) {
    if (SEGMENTED_FIELDS.has(fieldName)) return "segmented";
    return "select";
  }

  if (schema.type === "boolean") return "toggle";
  if (schema.type === "integer" || schema.type === "number") return "number";
  if (schema.type === "array" && schema.items?.type === "number")
    return "array-number";
  if (schema.type === "string") return "text";

  return "skip";
}
