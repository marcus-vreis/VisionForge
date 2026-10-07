/** The sentence ModelAdvice shows for what /api/model/defaults found.
 *
 * The server also sends a `note`, but it is written in Portuguese; the same
 * cases are worded here, in the language of `t` (`const t = useT()`), from the
 * numbers the response carries. The conditions are the route's own: the
 * collapse warning first, else the upscaling one when the dataset's median side
 * is under the suggested size, else the plain suggested setting.
 *
 * The collapse warning says only what was measured (ADR-099/100). Its accuracy
 * comes from `collapse_evidence`, never from the wording. The suggested setting
 * is only said to train when `recovered_accuracy` carries the number that shows
 * it. The three ways of having no evidence are told apart:
 *   - `null`: a server that looked and found the family was never run, so the
 *     note says so (`unmeasured`);
 *   - absent, or an outcome this build cannot word: nothing is known, so the
 *     note is the plain suggestion and claims nothing about the family;
 *   - present: the measured sentence.
 *
 * Every number was measured on classification. On a regression or segmentation
 * form the note says so, and never carries the recovery over.
 */

import type { CollapseEvidence, ModelDefaults } from "../api/client";
import type { Dict, ModelAdviceTask } from "../i18n/pt";

export type { ModelAdviceTask };

/** The evidence, if it describes an outcome this build knows how to word. */
function knownEvidence(advice: ModelDefaults): CollapseEvidence | null {
  const evidence = advice.collapse_evidence;
  if (evidence?.outcome === "collapse" || evidence?.outcome === "fails_to_learn") return evidence;
  return null;
}

/** Whether ModelAdvice should raise its alarm. Only a measured failure on this
 *  task does: a flagged family with no evidence still gets the suggestion, in
 *  the milder style of the other notes, and so does a regression or
 *  segmentation form, whose evidence belongs to classification. */
export function isAlarming(advice: ModelDefaults, task: ModelAdviceTask): boolean {
  return task === "classification" && advice.collapse_prone && knownEvidence(advice) !== null;
}

export function modelAdviceNote(t: Dict, advice: ModelDefaults, task: ModelAdviceTask): string {
  const words = t.modelAdvice;
  const { architecture, optimizer, learning_rate: rate } = advice;
  if (advice.collapse_prone) {
    if (advice.collapse_evidence === null) return words.unmeasured(architecture, optimizer, rate);
    const evidence = knownEvidence(advice);
    // Absent (an older server) or an outcome this build cannot word: claim nothing.
    if (!evidence) return words.suggested(architecture, optimizer, rate);
    const describe =
      evidence.outcome === "fails_to_learn" ? words.failsToLearnMeasured : words.collapseMeasured;
    return describe({
      architecture,
      measuredOn: evidence.measured_on,
      accuracy: evidence.accuracy,
      recoveredAccuracy: evidence.recovered_accuracy ?? null,
      optimizer,
      learningRate: rate,
      task,
    });
  }
  const { dataset_median_side: median, image_size: size } = advice;
  if (median !== null && size !== null && median < size) {
    return words.upscaling(median);
  }
  return words.suggested(architecture, optimizer, rate);
}
