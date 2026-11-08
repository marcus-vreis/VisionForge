import type { CodeSettings } from "../api/client";

/**
 * The settings of a classification run that choose a branch in its code
 * (ADR-173), read from the submitted payload.
 *
 * A setting is left out when the payload does not decide it, so its notes
 * stay undecided instead of naming a path the run may not take. Mixed
 * precision is the case: the trainer turns it on only when the device really
 * is CUDA (`use_amp = mixed_precision and device.type == "cuda"`), and an
 * `auto` or `cuda` choice can still fall back to the CPU.
 */
export function codeSettingsFor(payload: Record<string, unknown>): CodeSettings {
  const settings: CodeSettings = {};
  const task = payload["task"];
  if (task === "binary" || task === "multiclass") settings.kind = task;

  const training = record(payload["training"]);
  const device = record(payload["device"]);
  if (training["mixed_precision"] === false || device["kind"] === "cpu") settings.amp = false;

  const transforms = record(payload["transforms"]);
  if (typeof transforms["augment"] === "boolean") settings.augment = transforms["augment"];

  const name = record(payload["model"])["name"];
  if (typeof name === "string" && name) settings.model = name;
  return settings;
}

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}
