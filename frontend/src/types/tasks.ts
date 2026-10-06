import type { Dict } from "../i18n/pt";

export interface TaskDefinition {
  key: string;
  label: string;
  short: string;
  description: string;
  /** Hex color used for tab indicator dots and overlay progress bar. */
  accent: string;
}

/** The five built-in tasks, with their text in the active language. Not a hook,
 *  so the dictionary comes in as an argument: `taskDefinitions(t)` with
 *  `const t = useT()`. Keys and accents are the same in every language; a
 *  researcher-defined task brings its own text (see lib/custom-tasks.ts). */
export function taskDefinitions(t: Dict): TaskDefinition[] {
  return [
    {
      key: "classification",
      label: t.tasks.classification.label,
      short: "class",
      description: t.tasks.classification.description,
      accent: "#f16363",
    },
    {
      key: "detection",
      label: t.tasks.detection.label,
      short: "detect",
      description: t.tasks.detection.description,
      accent: "#48cf8e",
    },
    {
      key: "regression",
      label: t.tasks.regression.label,
      short: "reg",
      description: t.tasks.regression.description,
      accent: "#5b9fff",
    },
    {
      key: "segmentation",
      label: t.tasks.segmentation.label,
      short: "seg",
      description: t.tasks.segmentation.description,
      accent: "#b079ff",
    },
    {
      key: "anomaly",
      label: t.tasks.anomaly.label,
      short: "anom",
      description: t.tasks.anomaly.description,
      accent: "#f5a524",
    },
  ];
}
