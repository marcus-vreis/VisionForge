/** "Primeiro treino": a run on a synthetic dataset, step by step (ADR-115, ADR-117).
 *
 * Eight stops, the same for every task. Three of them wait for the researcher: the
 * dataset field filled, the training started, the training finished. The first one
 * offers a button that creates the sample dataset and fills the field, so the guide
 * can be followed without owning a single image.
 *
 * The script is played on the tab that is active when the guide opens: what the data
 * is, the parameters that matter, what to watch and the result metric are that
 * task's own (`guides.firstTraining.tasks.<task>`); the gates, the anchors and the
 * wording of the buttons are shared.
 */

import { createSampleDataset } from "../../api/client";
import type { Dict } from "../../i18n/pt";
import type { GuideTask } from "./tasks";
import type { GuideDefinition, GuideStep, SampleSetup } from "./types";

/** The last two segments of a path ("datasets/exemplo-classificacao"): the card names
 *  the folder, and the dataset field above it already shows the whole path. */
export function folderTail(path: string): string {
  return path.split(/[\\/]/).filter(Boolean).slice(-2).join("/");
}

/** What each task's sample fixes in its form (see `SampleSetup`). */
export const SAMPLE_SETUP: Record<GuideTask, SampleSetup> = {
  classification: {},
  detection: { numClasses: 1, imageSize: 128 },
  regression: { imageSize: 64 },
  segmentation: { numClasses: 3, imageSize: 64 },
  anomaly: { imageSize: 64 },
};

function firstTrainingSteps(t: Dict, task: GuideTask = "classification"): GuideStep[] {
  const s = t.guides.firstTraining;
  const x = s.tasks[task];
  return [
    {
      anchor: "dataset",
      align: "top",
      title: x.sample.title,
      body: x.sample.body,
      onEnter: (ctx) => ctx.selectTask(task),
      action: {
        label: s.action,
        run: async (ctx) => {
          try {
            const made = await createSampleDataset(task);
            ctx.setDatasetPath(made.path, task, SAMPLE_SETUP[task]);
            return {
              ok: true,
              message: made.existed
                ? s.existed(folderTail(made.path))
                : s.created(folderTail(made.path)),
            };
          } catch (e) {
            return {
              ok: false,
              message: s.failed(e instanceof Error ? e.message : String(e)),
            };
          }
        },
      },
    },
    {
      // Classification's picker shows what it found inside the highlighted box; the
      // other panels show it in a stats card below their folder row, which the
      // guide's card would otherwise sit on.
      anchor: task === "classification" ? "dataset" : "dataset-stats",
      align: "top",
      title: x.dataset.title,
      body: x.dataset.body,
      waitFor: (facts) => facts.datasetPath.trim() !== "",
      waitHint: s.datasetWaiting,
    },
    {
      anchor: "training",
      align: "top",
      title: x.parameters.title,
      body: x.parameters.body,
    },
    {
      anchor: "train",
      title: x.train.title,
      body: x.train.body,
      waitFor: (facts) => facts.trainingStarted,
      waitHint: s.trainWaiting,
    },
    {
      anchor: "training-sheet",
      floating: true,
      title: x.watching.title,
      body: x.watching.body,
    },
    {
      anchor: "training-sheet",
      floating: true,
      title: x.result.title,
      body: x.result.body,
      waitFor: (facts) => facts.trainingEnded,
      waitHint: s.resultWaiting,
    },
    {
      anchor: "history",
      title: s.history.title,
      body: s.history.body,
      // The sheet covers the bottom bar; the step points at a button under it.
      onEnter: (ctx) => ctx.hideTrainingSheet(),
    },
    {
      anchor: "datasets",
      title: x.others.title,
      body: x.others.body,
    },
  ];
}

export const firstTrainingGuide: GuideDefinition = {
  id: "firstTraining",
  title: (t) => t.guides.firstTraining.title,
  summary: (t) => t.guides.firstTraining.summary,
  steps: firstTrainingSteps,
};
