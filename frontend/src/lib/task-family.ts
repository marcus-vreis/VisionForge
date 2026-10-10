import type { Dict } from "../i18n/pt";

/** Order the family tabs the way the app's own task bar orders them, so the
 * history reads like the rest of the GUI instead of alphabetically. Custom
 * tasks (ADR-058) keep their own key and come after these. */
export const FAMILY_ORDER = [
  "classification",
  "detection",
  "regression",
  "segmentation",
  "anomaly",
];

/** Map a run's `task` onto the family it belongs to.
 *
 * `run.task` is not the family: classification runs record their *problem*
 * type (`binary`, `multiclass`, `multilabel`) because that is what the
 * classification config's `task` field means, while the standalone tasks
 * record the family itself. Grouping on the raw value split classification
 * into a "BINARY" and a "MULTICLASS" tab, which is not a task anyone chose.
 */
export function taskFamily(task: string): string {
  if (task.startsWith("custom:")) return task;
  if (FAMILY_ORDER.includes(task) && task !== "classification") return task;
  return "classification";
}

/** Tab label per task family, from the dictionary; a custom task is its own name. */
export function familyLabel(t: Dict, family: string): string {
  if (family.startsWith("custom:")) return family.slice("custom:".length);
  const families: Record<string, string> = t.taskNames;
  return families[family] ?? family;
}
