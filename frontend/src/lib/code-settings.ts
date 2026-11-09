import type { CodeSettings } from "../api/client";

/**
 * The settings of a run that choose a branch in its code (ADR-173), read from
 * the submitted payload of `taskKey`.
 *
 * A setting is left out when the payload does not decide it, so its notes
 * stay undecided instead of naming a path the run may not take. Mixed
 * precision is the case: the trainer turns it on only when the device really
 * is CUDA (`use_amp = mixed_precision and device.type == "cuda"`), and an
 * `auto` or `cuda` choice can still fall back to the CPU. The server drops a
 * value the task does not take.
 */
export function codeSettingsFor(taskKey: string, payload: Record<string, unknown>): CodeSettings {
  if (taskKey === "classification") return classificationSettings(payload);
  if (taskKey === "regression") return lossSettings(payload);
  if (taskKey === "segmentation") return { ...lossSettings(payload), ...builtModel(payload) };
  if (taskKey === "anomaly") return anomalySettings(payload);
  if (taskKey === "detection") return detectionSettings(payload);
  return {};
}

/** `model.backend` and `model.name` of a detection run: only the torchvision
 *  backend trains with this project's code, and the name picks its builder. */
function detectionSettings(payload: Record<string, unknown>): CodeSettings {
  const model = record(payload["model"]);
  const settings: CodeSettings = {};
  const backend = model["backend"];
  if (backend === "torchvision" || backend === "ultralytics") settings.backend = backend;
  const name = model["name"];
  if (typeof name === "string" && name) settings.model = name;
  return settings;
}

/** `model.name` of an anomaly run is its method: the autoencoder or PatchCore. */
function anomalySettings(payload: Record<string, unknown>): CodeSettings {
  const name = record(payload["model"])["name"];
  return name === "autoencoder" || name === "patchcore" ? { method: name } : {};
}

/** The model name the factory builds from; a model of the researcher's own
 *  (`custom_model`) skips that code, so it decides nothing. */
function builtModel(payload: Record<string, unknown>): CodeSettings {
  const model = record(payload["model"]);
  const custom = model["custom_model"];
  if (typeof custom === "string" && custom) return {};
  const name = model["name"];
  return typeof name === "string" && name ? { model: name } : {};
}

function classificationSettings(payload: Record<string, unknown>): CodeSettings {
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

/** `training.loss`, the criterion of a regression or segmentation run. */
function lossSettings(payload: Record<string, unknown>): CodeSettings {
  const loss = record(payload["training"])["loss"];
  return typeof loss === "string" && loss ? { loss } : {};
}

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}
