/** Shared helpers to rebuild a panel form from an imported YAML payload
 *  (ADR-059 experiment-header contract). The form defaults drive the shape:
 *  a payload value is taken only when its primitive type matches the default's,
 *  so a malformed YAML degrades to defaults instead of corrupting the form.
 *  Task-specific mismatches (arrays↔strings, sentinels) are handled by small
 *  adapters in each task's models lib. */

import type { PreprocessingStep } from "../components/PreprocessingPanel";
import {
  makeDefaultTransformsForm,
  type TransformsForm,
} from "./transforms-form";

function isPlainObject(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

/** Deep-merge `source` into a copy of `defaults`, keyed by the DEFAULTS shape:
 *  unknown source keys are ignored, type-mismatched leaves keep the default. */
export function mergeFormShape<T>(defaults: T, source: unknown): T {
  if (isPlainObject(defaults)) {
    const src = isPlainObject(source) ? source : {};
    const out: Record<string, unknown> = {};
    for (const [key, defVal] of Object.entries(defaults)) {
      out[key] = mergeFormShape(defVal, src[key]);
    }
    return out as T;
  }
  if (Array.isArray(defaults)) {
    return (Array.isArray(source) ? source : defaults) as T;
  }
  if (typeof source === typeof defaults && source !== null) {
    return source as T;
  }
  return defaults;
}

/** Join a payload list into the comma-separated string the UI edits. */
export function joinList(value: unknown): string | null {
  if (!Array.isArray(value) || value.length === 0) return null;
  return value.map(String).join(", ");
}

/** Rebuild the augmentation/normalization form from a `data.transforms`
 *  payload node (normalize triples become comma-separated strings). */
export function transformsFormFromPayload(node: unknown): TransformsForm {
  const defaults = makeDefaultTransformsForm();
  if (!isPlainObject(node)) return defaults;
  return {
    horizontal_flip:
      typeof node.horizontal_flip === "boolean"
        ? node.horizontal_flip
        : defaults.horizontal_flip,
    rotation_degrees:
      typeof node.rotation_degrees === "number"
        ? node.rotation_degrees
        : defaults.rotation_degrees,
    color_jitter:
      typeof node.color_jitter === "boolean"
        ? node.color_jitter
        : defaults.color_jitter,
    normalize_mean: joinList(node.normalize_mean) ?? defaults.normalize_mean,
    normalize_std: joinList(node.normalize_std) ?? defaults.normalize_std,
  };
}

/** Rebuild UI preprocessing steps from a `data.preprocessing` payload node
 *  (schema-flat `{kind, ...params}` → `{kind, params}`). */
export function stepsFromPayload(node: unknown): PreprocessingStep[] {
  if (!isPlainObject(node) || !Array.isArray(node.steps)) return [];
  const steps: PreprocessingStep[] = [];
  for (const raw of node.steps) {
    if (!isPlainObject(raw) || typeof raw.kind !== "string") continue;
    const { kind, ...params } = raw;
    steps.push({
      kind,
      params: Object.fromEntries(
        Object.entries(params).filter(
          ([, v]) => typeof v === "string" || typeof v === "number",
        ),
      ) as Record<string, string | number>,
    });
  }
  return steps;
}

/** Map a `transfer_learning` payload node to the form's mode + multiplier. */
export function transferFromPayload(node: unknown): {
  transfer: "none" | "feature_extraction" | "fine_tuning";
  backbone_lr_multiplier: number | null;
} {
  if (!isPlainObject(node)) return { transfer: "none", backbone_lr_multiplier: null };
  const mode = node.mode;
  if (mode === "feature_extraction") {
    return { transfer: "feature_extraction", backbone_lr_multiplier: null };
  }
  if (mode === "fine_tuning") {
    return {
      transfer: "fine_tuning",
      backbone_lr_multiplier:
        typeof node.backbone_lr_multiplier === "number"
          ? node.backbone_lr_multiplier
          : null,
    };
  }
  return { transfer: "none", backbone_lr_multiplier: null };
}
