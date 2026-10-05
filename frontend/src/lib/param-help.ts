/** One line per hyperparameter, and which ones start collapsed.
 *
 * The request that produced this was "conheço apenas metade desses; tá bem
 * difícil de entender, muita coisa junta" — and, explicitly, *not* to remove
 * any. So nothing here hides a parameter: the advanced ones are one click away
 * and travel in the payload with the same value as always.
 *
 * The split is by **how often a value changes**, not by how important it is.
 * The optimizer matters enormously and is still advanced, because it gets
 * decided once and then left alone; epochs and learning rate are what move
 * between one experiment and the next.
 *
 * The explanations themselves live in the language dictionaries (`paramHelp`
 * in src/i18n), one per language, and the test beside this file checks that
 * every classified field has one in each. Keeping them as data rather than
 * scattered through JSX is what makes "every field is explained" a test rather
 * than a promise.
 */

import type { Dict } from "../i18n/pt";

export type ParamTier = "basic" | "advanced";

/** Which tier each parameter belongs to. Anything absent counts as basic. */
export const PARAM_TIER: Record<string, ParamTier> = {
  epochs: "basic",
  batch_size: "basic",
  learning_rate: "basic",
  seed: "basic",

  optimizer: "advanced",
  momentum: "advanced",
  weight_decay: "advanced",
  learning_rate_final: "advanced",
  lrf: "advanced",
  scheduler: "advanced",
  step_size: "advanced",
  gamma: "advanced",
  cos_lr: "advanced",
  warmup_epochs: "advanced",
  early_stopping_patience: "advanced",
  patience: "advanced",
  label_smoothing: "advanced",
  dropout: "advanced",
  freeze: "advanced",
  amp: "advanced",
  deterministic: "advanced",
  mixed_precision: "advanced",
  num_workers: "advanced",
  workers: "advanced",
  pin_memory: "advanced",
  nbs: "advanced",
  single_cls: "advanced",
  rect: "advanced",
  multi_scale: "advanced",
  close_mosaic: "advanced",
  box: "advanced",
  cls: "advanced",
  dfl: "advanced",
};

/** Whether a parameter starts collapsed.
 *
 * An unclassified name counts as basic: a field nobody thought about must stay
 * visible rather than disappear by accident.
 */
export function isAdvanced(key: string): boolean {
  return PARAM_TIER[key] === "advanced";
}

/** The explanation for a parameter in the active language, or undefined if it
 * has none yet. Not a hook, so the dictionary comes in as an argument:
 * `paramHelp(t, "epochs")` with `const t = useT()`.
 *
 * `key` is a backend field name, or the dot-path of a field whose meaning
 * depends on where it sits (`training.scheduler.patience`). The dictionary pins
 * the set of keys per language; here they are widened to open strings because
 * the caller's key is not known at compile time. */
export function paramHelp(t: Dict, key: string): string | undefined {
  const help: Record<string, string> = t.paramHelp;
  return help[key];
}

/** Whether any advanced field differs from its default.
 *
 * The advanced section starts collapsed, but hiding a value the researcher
 * deliberately set — or that arrived with an imported YAML — would be worse
 * than the clutter the collapsing exists to remove. So a tuned form opens.
 */
export function hasNonDefaultAdvanced(
  form: Record<string, unknown>,
  defaults: Record<string, unknown>,
): boolean {
  return Object.keys(form).some((key) => {
    if (!isAdvanced(key)) return false;
    // A default we do not know is not evidence of a change. Nested objects
    // (the scheduler) carry no `default` at their own level in the schema, and
    // counting them as different made the section open every single time —
    // which is the same as not having a section.
    if (defaults[key] === undefined) return false;
    return JSON.stringify(form[key]) !== JSON.stringify(defaults[key]);
  });
}
