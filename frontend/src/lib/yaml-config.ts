import * as jsyaml from "js-yaml";
import type { Dict } from "../i18n/pt";
import type { JsonSchema } from "../types/schema";
import type { ValidationError } from "../hooks/useExperiment";

/**
 * Serialize formData to YAML and trigger a browser file download.
 * Filename format: experiment_<name>_<YYYY-MM-DD>.yaml
 */
export function exportConfigToYaml(
  formData: Record<string, unknown>,
  experimentName: string,
): void {
  const yaml = serializeConfigToYaml(formData);
  const date = new Date().toISOString().slice(0, 10);
  const safeName = (experimentName || "config").replace(/[^\w-]/g, "_");
  const filename = `experiment_${safeName}_${date}.yaml`;

  const blob = new Blob([yaml], { type: "text/yaml;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

/**
 * Read a File, parse it as YAML, and return either the parsed config object
 * or an error message in the language of `t`.
 */
export async function importConfigFromYaml(
  t: Dict,
  file: File,
): Promise<{ data: Record<string, unknown> } | { error: string }> {
  let text: string;
  try {
    text = await file.text();
  } catch (e) {
    const reason = e instanceof Error ? e.message : String(e);
    return { error: t.yamlConfig.cannotRead(reason) };
  }

  try {
    const data = parseYamlToConfig(text);
    return { data };
  } catch (e) {
    if (e instanceof YamlParseError) {
      // A syntax error keeps js-yaml's own message (it names the line); the
      // other kind is VisionForge's, so it is worded from the dictionary.
      const reason = e.kind === "notMapping" ? t.yamlConfig.notMapping : e.message;
      return { error: t.yamlConfig.invalidFile(reason) };
    }
    const reason = e instanceof Error ? e.message : String(e);
    return { error: t.yamlConfig.cannotRead(reason) };
  }
}

/**
 * "syntax": the text is not valid YAML; `message` is js-yaml's own.
 * "notMapping": valid YAML whose root is a scalar or a list, not key-value
 * pairs; the UI words it from the dictionary, `message` is for developers.
 */
export type YamlParseErrorKind = "syntax" | "notMapping";

export class YamlParseError extends Error {
  readonly kind: YamlParseErrorKind;

  constructor(kind: YamlParseErrorKind, message: string) {
    super(message);
    this.name = "YamlParseError";
    this.kind = kind;
  }
}

/** Recursively drop `undefined` values; preserve `null` as-is. */
export function sanitizeForExport(data: unknown): unknown {
  if (data === null || data === undefined) return data === undefined ? undefined : null;
  if (Array.isArray(data)) return data.map(sanitizeForExport);
  if (typeof data === "object") {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(data as Record<string, unknown>)) {
      if (v !== undefined) out[k] = sanitizeForExport(v);
    }
    return out;
  }
  return data;
}

/** Serialize form data to a YAML string matching baseline.yaml conventions. */
export function serializeConfigToYaml(formData: Record<string, unknown>): string {
  const clean = sanitizeForExport(formData) as Record<string, unknown>;
  return jsyaml.dump(clean, { indent: 2, lineWidth: -1 });
}

/**
 * Parse a YAML string into a config object.
 * Uses js-yaml's default safe loader (v4). Never uses loadAll or unsafe schemas.
 * Throws YamlParseError on malformed input or non-object root.
 */
export function parseYamlToConfig(yamlText: string): Record<string, unknown> {
  let parsed: unknown;
  try {
    // js-yaml v4 load() is safe by default — no code execution, no unsafe types
    parsed = jsyaml.load(yamlText);
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    throw new YamlParseError("syntax", msg);
  }
  if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new YamlParseError(
      "notMapping",
      "YAML must contain a mapping (key-value object), not a scalar or list.",
    );
  }
  return parsed as Record<string, unknown>;
}

/**
 * Validate parsed config data against a JSON Schema.
 * Checks shape, required fields, and basic leaf types only.
 * Cross-field invariants (e.g. task ↔ num_classes) are intentionally excluded —
 * those are enforced server-side via Pydantic and surfaced through the 422 path.
 * The messages are worded in the language of `t`.
 */
export function validateParsedConfig(
  t: Dict,
  data: unknown,
  schema: JsonSchema,
  defs: Record<string, JsonSchema> = {},
  path: string[] = [],
): ValidationError[] {
  const errors: ValidationError[] = [];
  const msg = t.yamlConfig;
  const resolved = resolveRef(schema, defs);

  if (resolved.anyOf) {
    // `Optional[X]`: an explicit null is a valid value, and the export writes
    // one out for every unset optional, so our own files must import clean.
    if (data === null && resolved.anyOf.some((s) => s.type === "null")) return errors;
    const nonNull = resolved.anyOf.find((s) => s.type !== "null");
    if (nonNull) return validateParsedConfig(t, data, nonNull, defs, path);
    return errors;
  }

  if (resolved.type === "object" && resolved.properties) {
    if (data === null || typeof data !== "object" || Array.isArray(data)) {
      errors.push({ field: path, message: msg.expectedObject });
      return errors;
    }
    const obj = data as Record<string, unknown>;
    for (const [key, propSchema] of Object.entries(resolved.properties)) {
      const childPath = [...path, key];
      if (!(key in obj)) {
        if (resolved.required?.includes(key)) {
          errors.push({ field: childPath, message: msg.requiredMissing });
        }
        continue;
      }
      errors.push(...validateParsedConfig(t, obj[key], propSchema, defs, childPath));
    }
    return errors;
  }

  if (resolved.enum) {
    if (!resolved.enum.includes(data as string | number | boolean)) {
      errors.push({
        field: path,
        message: msg.mustBeOneOf(resolved.enum.map(String).join(", ")),
      });
    }
    return errors;
  }

  if (resolved.type === "boolean" && typeof data !== "boolean") {
    errors.push({ field: path, message: msg.expectedBoolean });
    return errors;
  }

  if (
    (resolved.type === "number" || resolved.type === "integer") &&
    typeof data !== "number"
  ) {
    errors.push({
      field: path,
      message: resolved.type === "integer" ? msg.expectedInteger : msg.expectedNumber,
    });
    return errors;
  }

  if (resolved.type === "string" && typeof data !== "string") {
    errors.push({ field: path, message: msg.expectedString });
    return errors;
  }

  return errors;
}

interface PathNode {
  /** The value at this path is what was flagged: remove it whole. */
  drop: boolean;
  children: Map<string, PathNode>;
}

/**
 * Remove from a parsed config every value that `validateParsedConfig` flagged,
 * so the form is never handed a leaf of the wrong type (a number where the
 * panels call `.trim()` on a string, say). A removed leaf is simply absent,
 * the same state as a field the YAML never mentioned, which every panel
 * already renders.
 *
 * Takes the issues' `field` paths rather than the schema so it stays a plain
 * tree edit. A path that points at nothing (a missing required field) is
 * skipped without conjuring the parents; arrays are left intact except for an
 * element a path names, which is dropped without leaving a hole. The input is
 * never mutated and anything untouched keeps its identity.
 *
 * `omitted` lists the paths actually removed, so the caller can tell the user.
 */
export function omitInvalidLeaves(
  data: Record<string, unknown>,
  issues: ReadonlyArray<{ field: readonly (string | number)[] }>,
): { data: Record<string, unknown>; omitted: string[][] } {
  const root: PathNode = { drop: false, children: new Map() };
  for (const { field } of issues) {
    if (field.length === 0) continue; // the root is the file itself; nothing to drop
    let node = root;
    for (const segment of field) {
      const key = String(segment);
      let next = node.children.get(key);
      if (!next) {
        next = { drop: false, children: new Map() };
        node.children.set(key, next);
      }
      node = next;
    }
    node.drop = true;
  }
  const omitted: string[][] = [];
  const pruned = prune(data, root, [], omitted) as Record<string, unknown>;
  return { data: pruned, omitted };
}

function prune(value: unknown, node: PathNode, at: string[], omitted: string[][]): unknown {
  if (node.children.size === 0) return value;
  const isList = Array.isArray(value);
  if (!isList && (value === null || typeof value !== "object")) return value;

  const entries: [string, unknown][] = isList
    ? value.map((v, i) => [String(i), v])
    : Object.entries(value as Record<string, unknown>);
  const kept: [string, unknown][] = [];
  let changed = false;
  for (const [key, child] of entries) {
    const sub = node.children.get(key);
    if (!sub) {
      kept.push([key, child]);
      continue;
    }
    const path = [...at, key];
    if (sub.drop) {
      omitted.push(path);
      changed = true;
      continue;
    }
    const next = prune(child, sub, path, omitted);
    if (next !== child) changed = true;
    kept.push([key, next]);
  }
  if (!changed) return value;
  // fromEntries defines own properties, so a hostile `__proto__` key in the
  // YAML stays an ordinary key instead of swapping the prototype.
  return isList ? kept.map(([, v]) => v) : Object.fromEntries(kept);
}

function resolveRef(schema: JsonSchema, defs: Record<string, JsonSchema>): JsonSchema {
  if (schema.$ref) {
    const name = schema.$ref.split("/").pop()!;
    return defs[name] ?? schema;
  }
  return schema;
}
