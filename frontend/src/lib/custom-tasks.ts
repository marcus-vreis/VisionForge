/**
 * Merging researcher-defined tasks into the tab bar (ADR-058).
 *
 * `GET /api/tasks` returns the five built-in descriptors plus every task
 * registered from `user_tasks/`. The built-ins keep their rich local
 * definitions (model lists, curated params); a custom task carries only what
 * its `@register_task` declared — key, label, accent, description, metric
 * metadata — because its whole form is generated from the Config's JSON
 * Schema at runtime. No user-supplied JavaScript, ever.
 */
import type { JsonSchema } from "../types/schema";
import type { TaskDefinition } from "../types/tasks";

/** One row of `GET /api/tasks`. */
export interface TaskDescriptor {
  key: string;
  label: string;
  accent: string;
  description: string;
  custom: boolean;
  metrics: Record<string, string>;
  primary_metric: string;
}

/** A task definition that also knows it came from `user_tasks/`. */
export interface CustomTaskDefinition extends TaskDefinition {
  custom: true;
  metrics: Record<string, string>;
  primaryMetric: string;
}

export function isCustomTask(
  task: TaskDefinition,
): task is CustomTaskDefinition {
  return (task as CustomTaskDefinition).custom === true;
}

/** Turn a custom descriptor into the shape TabBar/TaskHero already render. */
export function descriptorToDefinition(d: TaskDescriptor): CustomTaskDefinition {
  return {
    key: d.key,
    label: d.label,
    // The hero's "short" tag: the key is more useful than a truncated label.
    short: d.key,
    description: d.description || "Tarefa definida pelo pesquisador",
    accent: d.accent,
    // Empty: a custom task has no curated model list or param cards — the
    // form comes from its schema.
    models: [],
    params: [],
    defaults: {},
    custom: true,
    metrics: d.metrics ?? {},
    primaryMetric: d.primary_metric,
  };
}

/**
 * Append custom descriptors to the built-in tabs, sorted by label.
 *
 * Built-ins keep their local definitions and their order — a custom task can
 * never displace or shadow one (the backend rejects those keys anyway), so a
 * broken user file cannot break the five tabs that always work.
 */
export function mergeTasks(
  builtins: TaskDefinition[],
  descriptors: TaskDescriptor[],
): TaskDefinition[] {
  const builtinKeys = new Set(builtins.map((t) => t.key));
  const customs = descriptors
    .filter((d) => d.custom && !builtinKeys.has(d.key))
    .sort((a, b) => a.label.localeCompare(b.label))
    .map(descriptorToDefinition);
  return [...builtins, ...customs];
}

/** Build a default form object from a JSON Schema, resolving $ref/anyOf. */
export function defaultsFromSchema(
  schema: JsonSchema,
  defs: Record<string, JsonSchema>,
): unknown {
  if (schema.$ref) {
    const name = schema.$ref.split("/").pop();
    const target = name ? defs[name] : undefined;
    return target ? defaultsFromSchema(target, defs) : null;
  }
  if (schema.default !== undefined) return schema.default;
  if (schema.anyOf) {
    const nonNull = schema.anyOf.find((s) => s.type !== "null");
    return nonNull ? defaultsFromSchema(nonNull, defs) : null;
  }
  if (schema.type === "object" && schema.properties) {
    const out: Record<string, unknown> = {};
    for (const [key, prop] of Object.entries(schema.properties)) {
      out[key] = defaultsFromSchema(prop, defs);
    }
    return out;
  }
  if (schema.enum?.length) return schema.enum[0];
  if (schema.type === "string") return "";
  if (schema.type === "integer" || schema.type === "number") return 0;
  if (schema.type === "boolean") return false;
  if (schema.type === "array") return [];
  return null;
}

/** Defaults for a whole custom-task Config schema. */
export function buildCustomForm(schema: JsonSchema): Record<string, unknown> {
  const defs = schema.$defs ?? {};
  return (defaultsFromSchema(schema, defs) ?? {}) as Record<string, unknown>;
}

/**
 * The payload a custom-task run expects: the form as-is, with the live device
 * selection injected (the DeviceSelector owns it globally, so the schema form
 * never renders a device card — same rule as the built-in panels).
 */
export function buildCustomPayload(
  form: Record<string, unknown>,
  device: { kind: string; gpu_ids: number[] | null },
): Record<string, unknown> {
  return { ...form, device: { kind: device.kind, gpu_ids: device.gpu_ids } };
}

/** Metric options for the sweep/replicates cards, from the task's declaration. */
export function metricOptions(
  task: CustomTaskDefinition,
): { value: string; label: string }[] {
  const names = Object.keys(task.metrics);
  const ordered = names.includes(task.primaryMetric)
    ? [task.primaryMetric, ...names.filter((n) => n !== task.primaryMetric)]
    : names;
  return ordered.map((name) => ({
    value: name,
    label: task.metrics[name] === "lower" ? `${name} ↓` : `${name} ↑`,
  }));
}
