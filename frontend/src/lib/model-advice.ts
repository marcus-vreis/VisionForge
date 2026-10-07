/** The sentence ModelAdvice shows for what /api/model/defaults found.
 *
 * The server also sends a `note`, but it is written in Portuguese; the same
 * cases are worded here, in the language of `t` (`const t = useT()`), from the
 * numbers the response carries. The conditions are the route's own: the
 * collapse warning first, else the upscaling one when the dataset's median side
 * is under the suggested size, else the plain measured setting.
 *
 * The collapse warning says only what was measured (ADR-099/100). Its accuracy
 * comes from `collapse_evidence`, never from the wording: a family that was not
 * run, or a response from a server that predates the field, gets the unmeasured
 * sentence and no number.
 */

import type { ModelDefaults } from "../api/client";
import type { Dict } from "../i18n/pt";

export function modelAdviceNote(t: Dict, advice: ModelDefaults): string {
  const words = t.modelAdvice;
  const { architecture, optimizer, learning_rate: rate } = advice;
  if (advice.collapse_prone) {
    const evidence = advice.collapse_evidence;
    if (evidence?.outcome === "collapse") {
      return words.collapseMeasured(
        architecture, evidence.measured_on, evidence.accuracy, optimizer, rate,
      );
    }
    if (evidence?.outcome === "fails_to_learn") {
      return words.failsToLearnMeasured(
        architecture, evidence.measured_on, evidence.accuracy, optimizer, rate,
      );
    }
    // No evidence, or an outcome this build does not know how to word.
    return words.unmeasured(architecture, optimizer, rate);
  }
  const { dataset_median_side: median, image_size: size } = advice;
  if (median !== null && size !== null && median < size) {
    return words.upscaling(median);
  }
  return words.measured(architecture, optimizer, rate);
}
