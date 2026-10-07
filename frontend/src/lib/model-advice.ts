/** The sentence ModelAdvice shows for what /api/model/defaults found.
 *
 * The server also sends a `note`, but it is written in Portuguese; the same
 * cases are worded here, in the language of `t` (`const t = useT()`), from the
 * numbers the response carries. The conditions are the route's own: the
 * collapse warning first, else the upscaling one when the dataset's median side
 * is under the suggested size, else the plain suggested setting.
 *
 * The collapse warning says only what was measured (ADR-099/100). Its accuracy
 * comes from `collapse_evidence`, never from the wording: a family that was not
 * run, or a response from a server that predates the field, gets the unmeasured
 * sentence and no number. The same goes for the suggested setting: it is only
 * said to train when `recovered_accuracy` carries the number that shows it.
 */

import type { CollapseEvidence, ModelDefaults } from "../api/client";
import type { Dict } from "../i18n/pt";

/** The evidence, if it describes an outcome this build knows how to word. */
function knownEvidence(advice: ModelDefaults): CollapseEvidence | null {
  const evidence = advice.collapse_evidence;
  if (evidence?.outcome === "collapse" || evidence?.outcome === "fails_to_learn") return evidence;
  return null;
}

/** Whether ModelAdvice should raise its alarm. Only a measured failure does: a
 *  flagged family with no evidence still gets the suggestion, in the milder
 *  style of the other notes. */
export function isAlarming(advice: ModelDefaults): boolean {
  return advice.collapse_prone && knownEvidence(advice) !== null;
}

export function modelAdviceNote(t: Dict, advice: ModelDefaults): string {
  const words = t.modelAdvice;
  const { architecture, optimizer, learning_rate: rate } = advice;
  if (advice.collapse_prone) {
    const evidence = knownEvidence(advice);
    // No evidence, or an outcome this build does not know how to word.
    if (!evidence) return words.unmeasured(architecture, optimizer, rate);
    const describe =
      evidence.outcome === "fails_to_learn" ? words.failsToLearnMeasured : words.collapseMeasured;
    return describe(
      architecture,
      evidence.measured_on,
      evidence.accuracy,
      evidence.recovered_accuracy ?? null,
      optimizer,
      rate,
    );
  }
  const { dataset_median_side: median, image_size: size } = advice;
  if (median !== null && size !== null && median < size) {
    return words.upscaling(median);
  }
  return words.suggested(architecture, optimizer, rate);
}
