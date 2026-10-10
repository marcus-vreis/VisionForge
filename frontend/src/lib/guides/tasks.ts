/** The tasks "Primeiro treino" has a script for (ADR-117): the five built-in tabs. */

export type GuideTask =
  | "classification"
  | "detection"
  | "regression"
  | "segmentation"
  | "anomaly";

export const GUIDE_TASKS: readonly GuideTask[] = [
  "classification",
  "detection",
  "regression",
  "segmentation",
  "anomaly",
];

/** The task a guide opened on the tab `activeKey` plays: that tab's own when it is
 *  a built-in task, classification for anything else (a researcher's own task has
 *  no sample dataset to train on). */
export function guideTaskFor(activeKey: string): GuideTask {
  return GUIDE_TASKS.find((task) => task === activeKey) ?? "classification";
}
