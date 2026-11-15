/** The label of a metric in a run's detail, from its name in run.json.
 *
 * Every task writes its own names (`test_r2`, `val_miou`, `image_f1`), and the
 * detail showed them raw for regression, segmentation and anomaly, because it
 * only knew the classification and detection ones. The names of all five tasks
 * are in the dictionaries already, for the History comparison
 * (`compareRuns.metrics`); the detail now reads them too, after its own.
 *
 * A researcher's own task reports whatever it likes: a name nothing here knows
 * is shown as it is.
 */

import type { Dict } from "../i18n/pt";

/** Metric names read the same in every language; only the words around them are translated. */
const METRIC_NAMES: Record<string, string> = {
  f1: "F1",
  recall: "Recall",
  auc_roc: "AUC-ROC",
  // Detection metrics (mAP @ IoU thresholds; box validation loss).
  map50: "mAP@50",
  map50_95: "mAP@50-95",
  box_loss: "Box loss (val)",
};

/** The metrics a classification test reports under a `test_` prefix. */
const TEST_METRICS = ["accuracy", "f1", "precision", "recall", "auc_roc"];

export function metricLabel(t: Dict, key: string): string {
  const words: Record<string, string> = t.runDetail.metrics.labels;
  const label = METRIC_NAMES[key] ?? words[key];
  if (label) return label;
  if (key.startsWith("test_") && TEST_METRICS.includes(key.slice(5))) {
    return t.runDetail.metrics.onTestSet(metricLabel(t, key.slice(5)));
  }
  const shared: Record<string, string> = t.compareRuns.metrics;
  return shared[key] ?? key;
}
