/** The experiment name the Revisão step suggests (ADR-155).
 *
 * The name used to be the first field of every panel, filled with "config" or
 * "detection_001", so History filled up with runs nobody could tell apart. The
 * suggestion is built from what sets a run apart — dataset, model, epochs,
 * learning rate — and, for a sweep, from what the sweep varies. The backend
 * then appends each trial's own value with the same short keys
 * (`src/visionforge/core/trial_name.py`), so a grid over the learning rate
 * lists `…_grid-lr_lr0.001`, `…_grid-lr_lr0.002`.
 */

/** Last segment of a dotted config path -> its short form in names. Mirrors
 *  SHORT_KEYS in core/trial_name.py. */
const SHORT_KEYS: Record<string, string> = {
  learning_rate: "lr",
  lr0: "lr",
  batch_size: "bs",
  weight_decay: "wd",
  epochs: "e",
  momentum: "mom",
  dropout: "drop",
  image_size: "img",
  imgsz: "img",
  optimizer: "opt",
  loss: "loss",
};

export function shortKey(dotKey: string): string {
  if (dotKey === "model.name") return "model";
  const last = dotKey.split(".").pop() ?? dotKey;
  return SHORT_KEYS[last] ?? last;
}

function slug(text: string): string {
  return text
    .normalize("NFD")
    .replace(/[̀-ͯ]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9.+-]+/g, "_")
    .replace(/^_+|_+$/g, "");
}

// Folder names that say where the data sits rather than what it is: the
// dataset is named by the folder above them.
const GENERIC_DIRS = new Set([
  "data",
  "dataset",
  "datasets",
  "images",
  "train",
  "val",
  "test",
]);

/** The dataset's name from its folder or its data.yaml path. */
export function datasetLabel(path: string): string {
  const parts = path
    .trim()
    .split(/[\\/]+/)
    .filter((p) => p !== "" && !/^[a-z]:$/i.test(p));
  if (parts.length === 0) return "";
  let last = parts[parts.length - 1].replace(/\.ya?ml$/i, "");
  if ((GENERIC_DIRS.has(last.toLowerCase()) || last === "data") && parts.length > 1) {
    last = parts[parts.length - 2];
  }
  return slug(last);
}

function formatNumber(value: number): string {
  // 0.0001 rather than 1e-4: the name is read, not parsed.
  if (Math.abs(value) > 0 && Math.abs(value) < 0.001) {
    return value.toFixed(10).replace(/0+$/, "");
  }
  return String(value);
}

export type NameStrategy =
  | "simple"
  | "cv"
  | "sweep"
  | "grid"
  | "random"
  | "replicates"
  | "compare"
  | "transfer";

export interface NameParts {
  /** Folder or data.yaml path of the dataset. */
  datasetPath: string;
  model?: string;
  epochs?: number;
  learningRate?: number;
  strategy?: NameStrategy;
  /** Dotted paths a sweep varies (grid, random, sweep). */
  sweepKeys?: string[];
  /** Number of folds, for cv. */
  folds?: number;
}

function strategyPart(p: NameParts): string | null {
  const keys = (p.sweepKeys ?? []).map(shortKey);
  switch (p.strategy) {
    case "grid":
    case "random":
    case "sweep": {
      const kind = p.strategy === "sweep" ? "sweep" : p.strategy;
      return keys.length > 0 ? `${kind}-${keys.join("-")}` : kind;
    }
    case "cv":
      return p.folds ? `cv${p.folds}` : "cv";
    case "replicates":
      return "rep";
    case "compare":
      return "compare";
    case "transfer":
      return "tl";
    default:
      return null;
  }
}

/** `<dataset>_<model>_e<epochs>_lr<lr>[_<strategy>]`. A value the sweep varies
 *  is left out of the base, since every trial adds its own. */
export function suggestExperimentName(p: NameParts): string {
  const varied = new Set((p.sweepKeys ?? []).map(shortKey));
  const parts: string[] = [];
  const dataset = datasetLabel(p.datasetPath);
  parts.push(dataset || "experimento");
  if (p.model && !varied.has("model")) parts.push(slug(p.model));
  if (p.epochs != null && Number.isFinite(p.epochs) && !varied.has("e")) {
    parts.push(`e${p.epochs}`);
  }
  if (
    p.learningRate != null &&
    Number.isFinite(p.learningRate) &&
    !varied.has("lr")
  ) {
    parts.push(`lr${formatNumber(p.learningRate)}`);
  }
  const strategy = strategyPart(p);
  if (strategy) parts.push(strategy);
  return parts.join("_");
}

// The last suggestion per task survives the panel unmounting when the
// researcher switches tabs, so coming back still tells a suggested name from
// one they typed.
const lastSuggestion = new Map<string, string>();

/** Whether `name` may be replaced by a new suggestion: empty, one of the
 *  task's placeholder defaults, or the previous suggestion. A name the
 *  researcher typed, or one a YAML import brought, stays. */
export function isAutoName(
  taskKey: string,
  name: string,
  defaults: readonly string[],
): boolean {
  return (
    name.trim() === "" ||
    defaults.includes(name) ||
    lastSuggestion.get(taskKey) === name
  );
}

export function rememberSuggestion(taskKey: string, suggestion: string): void {
  lastSuggestion.set(taskKey, suggestion);
}

/** Test hook: forget the remembered suggestions. */
export function resetSuggestions(): void {
  lastSuggestion.clear();
}
