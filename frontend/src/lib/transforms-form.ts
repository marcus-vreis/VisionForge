/** Shared augmentation/normalization form state for the standalone task panels
 *  (regression, segmentation, anomaly) — mirrors the Python `TransformConfig`
 *  in visionforge/utils/config.py, which those tasks' `data.transforms` reuse.
 *
 *  Defaults MUST equal the backend defaults: they were already silently applied
 *  to every GUI run of these tasks before the panels exposed them (ADR-059).
 *  Surfacing them unchanged is behavior-preserving; flipping them is the
 *  user's call now that they can see them. */

import type { PreprocessingStep } from "../components/PreprocessingPanel";

export interface TransformsForm {
  /** Off skips flip/rotation/jitter at training time without discarding them. */
  augment: boolean;
  horizontal_flip: boolean;
  rotation_degrees: number;
  color_jitter: boolean;
  /** Comma-separated triples in the UI; parsed into number lists on submit. */
  normalize_mean: string;
  normalize_std: string;
}

export function makeDefaultTransformsForm(): TransformsForm {
  return {
    augment: true,
    horizontal_flip: true,
    rotation_degrees: 10,
    color_jitter: false,
    normalize_mean: "0.485, 0.456, 0.406",
    normalize_std: "0.229, 0.224, 0.225",
  };
}

/** Parse a comma-separated float triple; fall back when malformed so a typo
 *  never produces a 422 for a field most users won't touch. */
export function parseTriple(raw: string, fallback: number[]): number[] {
  const parts = raw
    .split(",")
    .map((p) => Number(p.trim()))
    .filter((n) => Number.isFinite(n));
  return parts.length === 3 ? parts : fallback;
}

/** Project the transforms form into the `data.transforms` wire payload.
 *  `image_size` stays owned by each task's own field (regression
 *  `data.image_size` UI field / segmentation & anomaly `data.image_size`),
 *  so it is passed in explicitly where the task keeps it under transforms. */
export function buildTransformsPayload(
  t: TransformsForm,
  imageSize?: number,
): Record<string, unknown> {
  const payload: Record<string, unknown> = {
    augment: t.augment,
    horizontal_flip: t.horizontal_flip,
    rotation_degrees: t.rotation_degrees,
    color_jitter: t.color_jitter,
    normalize_mean: parseTriple(t.normalize_mean, [0.485, 0.456, 0.406]),
    normalize_std: parseTriple(t.normalize_std, [0.229, 0.224, 0.225]),
  };
  if (imageSize !== undefined) payload.image_size = imageSize;
  return payload;
}

/** Project UI preprocessing steps into the `data.preprocessing` wire payload —
 *  the backend `PreprocessStep` is schema-flat (`{kind, ...params}`). */
export function buildPreprocessingPayload(
  steps: PreprocessingStep[],
): Record<string, unknown> {
  return { steps: steps.map((s) => ({ kind: s.kind, ...s.params })) };
}
