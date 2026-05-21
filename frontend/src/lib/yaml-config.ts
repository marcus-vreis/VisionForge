import * as jsyaml from "js-yaml";
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
 * or a PT-BR friendly error message string.
 */
export async function importConfigFromYaml(
  file: File,
): Promise<{ data: Record<string, unknown> } | { error: string }> {
  let text: string;
  try {
    text = await file.text();
  } catch (e) {
    const reason = e instanceof Error ? e.message : String(e);
    return { error: `Não foi possível ler o arquivo YAML: ${reason}` };
  }

  try {
    const data = parseYamlToConfig(text);
    return { data };
  } catch (e) {
    if (e instanceof YamlParseError) {
      return { error: `Arquivo YAML inválido: ${e.message}` };
    }
    const reason = e instanceof Error ? e.message : String(e);
    return { error: `Não foi possível ler o arquivo YAML: ${reason}` };
  }
}

export class YamlParseError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "YamlParseError";
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
    throw new YamlParseError(`YAML parse error: ${msg}`);
  }
  if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new YamlParseError(
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
 */
export function validateParsedConfig(
  data: unknown,
  schema: JsonSchema,
  defs: Record<string, JsonSchema> = {},
  path: string[] = [],
): ValidationError[] {
  const errors: ValidationError[] = [];
  const resolved = resolveRef(schema, defs);

  if (resolved.anyOf) {
    const nonNull = resolved.anyOf.find((s) => s.type !== "null");
    if (nonNull) return validateParsedConfig(data, nonNull, defs, path);
    return errors;
  }

  if (resolved.type === "object" && resolved.properties) {
    if (data === null || typeof data !== "object" || Array.isArray(data)) {
      errors.push({ field: path, message: "Expected an object." });
      return errors;
    }
    const obj = data as Record<string, unknown>;
    for (const [key, propSchema] of Object.entries(resolved.properties)) {
      const childPath = [...path, key];
      if (!(key in obj)) {
        if (resolved.required?.includes(key)) {
          errors.push({ field: childPath, message: "Required field is missing." });
        }
        continue;
      }
      errors.push(...validateParsedConfig(obj[key], propSchema, defs, childPath));
    }
    return errors;
  }

  if (resolved.enum) {
    if (!resolved.enum.includes(data as string | number | boolean)) {
      errors.push({
        field: path,
        message: `Must be one of: ${resolved.enum.map(String).join(", ")}.`,
      });
    }
    return errors;
  }

  if (resolved.type === "boolean" && typeof data !== "boolean") {
    errors.push({ field: path, message: "Expected a boolean." });
    return errors;
  }

  if (
    (resolved.type === "number" || resolved.type === "integer") &&
    typeof data !== "number"
  ) {
    errors.push({ field: path, message: `Expected a ${resolved.type}.` });
    return errors;
  }

  if (resolved.type === "string" && typeof data !== "string") {
    errors.push({ field: path, message: "Expected a string." });
    return errors;
  }

  return errors;
}

function resolveRef(schema: JsonSchema, defs: Record<string, JsonSchema>): JsonSchema {
  if (schema.$ref) {
    const name = schema.$ref.split("/").pop()!;
    return defs[name] ?? schema;
  }
  return schema;
}
