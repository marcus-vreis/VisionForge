/** Which model a classification form trains when it is not the builtin one.
 *
 * `model.custom_model` (a builder in `user_models/`) and `model.timm_model`
 * replace `model.name` when the run is built (utils/config.py). The form has no
 * field for either: they arrive in an imported YAML, and the screens kept
 * naming the builtin architecture while another model trained.
 */

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
