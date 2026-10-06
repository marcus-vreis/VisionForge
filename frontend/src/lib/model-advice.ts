/** The sentence ModelAdvice shows for what /api/model/defaults found.
 *
 * The server also sends a `note`, but it is written in Portuguese; the same two
 * cases are worded here, in the language of `t` (`const t = useT()`), from the
 * numbers the response carries. The conditions are the route's own: the
 * collapse warning first, else the upscaling one when the dataset's median side
 * is under the suggested size, else the plain measured setting.
 */

import type { ModelDefaults } from "../api/client";
import type { Dict } from "../i18n/pt";

export function modelAdviceNote(t: Dict, advice: ModelDefaults): string {
  const words = t.modelAdvice;
  if (advice.collapse_prone) {
    return words.collapse(advice.architecture, advice.optimizer, advice.learning_rate);
  }
  const { dataset_median_side: median, image_size: size } = advice;
  if (median !== null && size !== null && median < size) {
    return words.upscaling(median);
  }
  return words.measured(advice.architecture, advice.optimizer, advice.learning_rate);
}
