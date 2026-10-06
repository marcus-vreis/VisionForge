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

  if (resolved.type === "object") {
    if (data === null || typeof data !== "object" || Array.isArray(data)) {
      errors.push({ field: path, message: msg.expectedObject });
      return errors;
    }
    const obj = data as Record<string, unknown>;
    const properties = resolved.properties ?? {};
    for (const [key, propSchema] of Object.entries(properties)) {
      const childPath = [...path, key];
      if (!(key in obj)) {
        if (resolved.required?.includes(key)) {
          errors.push({ field: childPath, message: msg.requiredMissing });
        }
        continue;
      }
      errors.push(...validateParsedConfig(t, obj[key], propSchema, defs, childPath));
    }
    // A dict field (`dict[str, X]`) has no properties of its own; what it holds
    // is checked against its value schema when the schema gives one.
    const entrySchema = dictEntrySchema(resolved, path);
    if (entrySchema) {
      for (const [key, value] of Object.entries(obj)) {
        if (Object.hasOwn(properties, key)) continue;
        errors.push(...validateParsedConfig(t, value, entrySchema, defs, [...path, key]));
      }
    }
    return errors;
  }

  if (resolved.type === "array") {
    if (!Array.isArray(data)) {
      errors.push({ field: path, message: msg.expectedArray });
      return errors;
    }
    const items = resolved.items;
    if (items) {
      data.forEach((item, index) => {
        errors.push(...validateParsedConfig(t, item, items, defs, [...path, String(index)]));
      });
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

/**
 * Dicts the schema leaves open (`dict[str, Any]`, so no value type to check)
 * whose entries the form nevertheless reads as mappings: each row of the
 * random-search editor takes `{type, low, high}` or `{type, options}` from its
 * entry, and a null there breaks the row. Keyed by the field's path from the
 * root of the config.
 */
const MAPPING_ENTRY_DICTS: ReadonlySet<string> = new Set(["random_search.search_space"]);

/** The schema each entry of a dict field must satisfy, or null when there is none to check. */
function dictEntrySchema(dict: JsonSchema, path: string[]): JsonSchema | null {
  const extra = dict.additionalProperties;
  if (extra !== null && typeof extra === "object") return extra;
  return MAPPING_ENTRY_DICTS.has(path.join(".")) ? { type: "object" } : null;
}

const PLAIN_NUMBER = /^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/;

/** The number a string spells out in plain decimal (what Pydantic reads), else
 *  null: not hex, not "NaN" or "Infinity", not "1_000", not a fraction for an integer. */
function numberFromString(text: string, integer: boolean): number | null {
  const trimmed = text.trim();
  if (!PLAIN_NUMBER.test(trimmed)) return null;
  const n = Number(trimmed);
  if (!Number.isFinite(n)) return null;
  if (integer && !Number.isInteger(n)) return null;
  return n;
}

/**
 * Read a quoted number as the backend does. Pydantic converts `"0.001"` to
 * 0.001 for a float field and `"10"` to 10 for an int one, so a file written
 * that way trains as intended; a validator stricter than that would drop the
 * value and swap in the default, which for a hyperparameter is worse than a
 * refusal. This turns such strings into numbers wherever the schema says a
 * number goes (a leaf, an item of a list, a value of a dict), and nothing else:
 * a string that is not plainly a number stays as written for the validator to
 * flag, and no other type is converted.
 *
 * Left alone on purpose: a field with a fixed set of values, and a field with
 * more than one non-null way to read it (`int | str`), where the string may be
 * what was meant. Returns the same object when nothing changed.
 */
export function coerceNumericStrings(
  data: Record<string, unknown>,
  schema: JsonSchema,
  defs: Record<string, JsonSchema> = {},
): Record<string, unknown> {
  return coerceValue(data, schema, defs) as Record<string, unknown>;
}

function coerceValue(value: unknown, schema: JsonSchema, defs: Record<string, JsonSchema>): unknown {
  const resolved = resolveRef(schema, defs);

  if (resolved.anyOf) {
    const branches = resolved.anyOf.filter((s) => s.type !== "null");
    return branches.length === 1 ? coerceValue(value, branches[0], defs) : value;
  }

  if (resolved.type === "number" || resolved.type === "integer") {
    if (typeof value !== "string" || resolved.enum || resolved.const !== undefined) return value;
    return numberFromString(value, resolved.type === "integer") ?? value;
  }

  if (resolved.type === "array") {
    const items = resolved.items;
    if (!Array.isArray(value) || !items) return value;
    const next = value.map((item) => coerceValue(item, items, defs));
    return next.some((item, i) => item !== value[i]) ? next : value;
  }

  if (resolved.type === "object") {
    if (value === null || typeof value !== "object" || Array.isArray(value)) return value;
    const properties = resolved.properties ?? {};
    const extra = resolved.additionalProperties;
    const entries = Object.entries(value as Record<string, unknown>).map(
      ([key, child]): [string, unknown] => {
        const sub = Object.hasOwn(properties, key)
          ? properties[key]
          : extra !== null && typeof extra === "object"
            ? extra
            : null;
        return [key, sub ? coerceValue(child, sub, defs) : child];
      },
    );
    const original = Object.values(value as Record<string, unknown>);
    // fromEntries, not assignment: a `__proto__` key stays an ordinary key.
    return entries.some(([, child], i) => child !== original[i]) ? Object.fromEntries(entries) : value;
  }

  return value;
}

/**
 * The import for the panels that refuse a file with problems instead of loading
 * what they can: quoted numbers are read as the backend reads them, then the
 * file either passes, with the data the form should take, or comes back as the
 * text of the refusal (the first five issues, one per line).
 */
export function checkImportedConfig(
  t: Dict,
  parsed: Record<string, unknown>,
  schema: JsonSchema,
  defs: Record<string, JsonSchema> = {},
): { data: Record<string, unknown> } | { problem: string } {
  const data = coerceNumericStrings(parsed, schema, defs);
  const issues = validateParsedConfig(t, data, schema, defs);
  if (issues.length === 0) return { data };
  return {
    problem: issues
      .slice(0, 5)
      .map((issue) => `${issue.field.join(" › ")}: ${issue.message}`)
      .join("\n"),
  };
}

/** What loading an imported file amounts to; see `reviewImportedConfig`. */
export interface ImportReview {
  /** The config to load: the parsed file minus every value that failed validation. */
  data: Record<string, unknown>;
  /** Everything validation found in the file as written, before anything was dropped. */
  issues: ValidationError[];
  /** Values of the wrong type left out whose field has a default to fall back to. */
  ignored: number;
  /** Required fields without a value in `data`: the file never set them, or the
   *  value it set was the wrong type and went out with the rest. */
  missing: number;
}

/**
 * Validate an imported file and prepare what the form may take. Quoted numbers
 * are read as the backend reads them (see `coerceNumericStrings`) before
 * anything is judged. A value of the wrong type is dropped (see
 * `omitInvalidLeaves`); the counts say what that leaves behind, so the warning
 * shown to the user can be exact about it: an optional field just uses its
 * default, a required one has to be filled in before the backend will accept
 * the config.
 */
export function reviewImportedConfig(
  t: Dict,
  parsed: Record<string, unknown>,
  schema: JsonSchema,
  defs: Record<string, JsonSchema> = {},
): ImportReview {
  const data = coerceNumericStrings(parsed, schema, defs);
  const issues = validateParsedConfig(t, data, schema, defs);
  const { data: kept, omitted } = omitInvalidLeaves(data, issues);
  // What is still flagged once the bad values are gone can only be absences.
  const stillMissing = validateParsedConfig(t, kept, schema, defs);
  const noValue = new Set(stillMissing.map((issue) => issue.field.join("\0")));
  return {
    data: kept,
    issues,
    ignored: omitted.filter((path) => !noValue.has(path.join("\0"))).length,
    missing: stillMissing.length,
  };
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
 * skipped without conjuring the parents. A list is all or nothing: when an
 * issue sits inside a list (an item, or a leaf of an item), the whole list goes
 * and the field falls back to its default, because a list with one item cut
 * out still passes the backend and then trains on values nobody wrote (a mean
 * of two channels, a GPU quietly lost). Lists nothing flagged are left intact.
 * The input is never mutated and anything untouched keeps its identity.
 *
 * `omitted` lists the paths actually removed, so the caller can tell the user.
 */
export function omitInvalidLeaves(
  data: Record<string, unknown>,
  issues: ReadonlyArray<{ field: readonly (string | number)[] }>,
): { data: Record<string, unknown>; omitted: string[][] } {
  const root: PathNode = { drop: false, children: new Map() };
  for (const { field } of issues) {
    const path = removalPath(data, field);
    if (path.length === 0) continue; // the root is the file itself; nothing to drop
    let node = root;
    for (const key of path) {
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

/**
 * What to remove for an issue at `field`: the nearest list the path runs
 * through (its items are what the issue is about, and the list stands or falls
 * as one), else the path itself. A path that does not resolve in `data` (a
 * required field that is absent) is returned as it is and removes nothing.
 */
function removalPath(data: unknown, field: readonly (string | number)[]): string[] {
  const path = field.map(String);
  let value = data;
  let list = -1;
  for (let i = 0; i < path.length; i++) {
    const key = path[i];
    if (Array.isArray(value)) {
      if (!/^\d+$/.test(key) || Number(key) >= value.length) return path;
      list = i;
      value = value[Number(key)];
    } else if (value !== null && typeof value === "object" && Object.hasOwn(value, key)) {
      value = (value as Record<string, unknown>)[key];
    } else {
      return path;
    }
  }
  return list >= 0 ? path.slice(0, list) : path;
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
