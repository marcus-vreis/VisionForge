/** What the Review says about the rate each trial trains at (ADR-158).
 *
 * The rates come from `POST /model/learning-rate-plan`, which calls the same
 * functions the blocks train with, so nothing here computes one. The warning
 * quotes only `collapse_evidence`: that measurement is one model per family at
 * Adam 1e-3, and the sentence says so instead of implying this combination ran.
 */
import type { LearningRatePlanRequest, TrialLearningRate } from "../api/client";
import type { Dict } from "../i18n/pt";

const section = (form: Record<string, unknown>, key: string) =>
  (form[key] ?? {}) as Record<string, unknown>;

/** The request for this form, or null while its rate is not a number yet. */
export function learningRatePlanRequest(
  form: Record<string, unknown>,
): LearningRatePlanRequest | null {
  const training = section(form, "training");
  const rate = training["learning_rate"];
  if (typeof rate !== "number") return null;
  const grid = section(form, "grid_search");
  const comparison = section(form, "model_comparison");
  return {
    block: String(form["block"] ?? "classification"),
    model_name: String(section(form, "model")["name"] ?? ""),
    optimizer: String(training["optimizer"] ?? "adam"),
    learning_rate: rate,
    hyperparameters: (grid["hyperparameters"] ?? {}) as Record<string, unknown[]>,
    lr_mode: grid["lr_mode"] === "suggested_multiplier" ? "suggested_multiplier" : "absolute",
    model_names: (comparison["model_names"] ?? []) as string[],
    lr_policy: comparison["lr_policy"] === "suggested" ? "suggested" : "fixed",
  };
}

/** Whether `training.learning_rate` of this form is not what the trials use. */
export function rateIsPerModel(form: Record<string, unknown>): boolean {
  const request = learningRatePlanRequest(form);
  if (!request) return false;
  return (
    (request.block === "grid_search" && request.lr_mode === "suggested_multiplier") ||
    (request.block === "model_comparison" && request.lr_policy === "suggested")
  );
}

/** The warning of a flagged combination; null when it is not flagged. */
export function collapseWarning(t: Dict, trial: TrialLearningRate): string | null {
  if (!trial.collapse_prone) return null;
  const words = t.stepShell.review.lrPlan;
  const combination = words.combination(trial.architecture, trial.optimizer, trial.learning_rate);
  const evidence = trial.collapse_evidence;
  if (evidence?.outcome !== "collapse" && evidence?.outcome !== "fails_to_learn") {
    return `${combination} ${words.notMeasured}`;
  }
  const finding = words.measured({
    measuredOn: evidence.measured_on,
    accuracy: evidence.accuracy,
    outcome: evidence.outcome,
  });
  return evidence.measured_on === trial.architecture.toLowerCase()
    ? `${combination} ${finding}`
    : `${combination} ${finding} ${words.siblingNotMeasured(trial.architecture)}`;
}
