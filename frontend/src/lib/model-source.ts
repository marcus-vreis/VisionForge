/** Which model a classification form trains when it is not the builtin one.
 *
 * `model.custom_model` (a builder in `user_models/`) and `model.timm_model`
 * replace `model.name` when the run is built (utils/config.py). The Model step
 * offers the custom models that `GET /api/model/custom` lists
 * (CustomModelField); `timm_model` has no field and arrives in an imported
 * YAML. The screens name whichever one trains instead of the builtin
 * architecture.
 */

import type { SelectOption } from "../components/controls/SelectField";

export type ModelSource = "custom_model" | "timm_model";

export interface AlternateModel {
  source: ModelSource;
  name: string;
}

const SOURCES: readonly ModelSource[] = ["custom_model", "timm_model"];

/** The model that replaces the builtin one, or null when the builtin trains.
 *  An empty or blank name means "not set", as the backend reads it. */
export function alternateModel(model: Record<string, unknown>): AlternateModel | null {
  for (const source of SOURCES) {
    const name = model[source];
    if (typeof name === "string" && name.trim()) return { source, name: name.trim() };
  }
  return null;
}

/** The custom-model selector's value for "the builtin architecture". */
export const BUILTIN_MODEL = "";

/** What the custom-model selector shows: the `custom_model` the form holds,
 *  trimmed, or `BUILTIN_MODEL` when none is set. */
export function customModelChoice(model: Record<string, unknown>): string {
  const name = model["custom_model"];
  return typeof name === "string" ? name.trim() : BUILTIN_MODEL;
}

/** `model` after picking `choice` in the selector. A custom model's name sets
 *  `custom_model` and clears `timm_model`, since the config refuses both at
 *  once; the builtin choice clears `custom_model`. */
export function withCustomModel(
  model: Record<string, unknown>,
  choice: string,
): Record<string, unknown> {
  const name = choice.trim();
  if (name === BUILTIN_MODEL) return { ...model, custom_model: null };
  return { ...model, custom_model: name, timm_model: null };
}

/** The selector's options: the builtin architecture first, then each model
 *  `user_models/` registers. A chosen name the folder does not register (an
 *  imported YAML can name one) stays listed, marked missing, so the field
 *  shows what the config holds. */
export function customModelOptions(
  registered: readonly string[],
  chosen: string,
  words: { builtin: string; builtinSub: string; missing: string },
): SelectOption[] {
  const options: SelectOption[] = [
    { value: BUILTIN_MODEL, label: words.builtin, sub: words.builtinSub },
    ...registered.map((name) => ({ value: name, label: name })),
  ];
  if (chosen !== BUILTIN_MODEL && !registered.includes(chosen)) {
    options.push({ value: chosen, label: chosen, sub: words.missing });
  }
  return options;
}
