/** "Primeiro treino": a classification run on a synthetic dataset, step by step (ADR-115).
 *
 * Eight stops. Three of them wait for the researcher: the dataset field filled,
 * the training started, the training finished. The first one offers a button
 * that creates the sample dataset and fills the field, so the guide can be
 * followed without owning a single image.
 */

import { createSampleDataset } from "../../api/client";
import type { Dict } from "../../i18n/pt";
import type { GuideDefinition, GuideStep } from "./types";

/** The last two segments of a path ("datasets/exemplo-classificacao"): the card names
 *  the folder, and the dataset field above it already shows the whole path. */
export function folderTail(path: string): string {
  return path.split(/[\\/]/).filter(Boolean).slice(-2).join("/");
}

function firstTrainingSteps(t: Dict): GuideStep[] {
  const s = t.guides.firstTraining;
  return [
    {
      anchor: "dataset",
      align: "top",
      title: s.sample.title,
      body: s.sample.body,
      onEnter: (ctx) => ctx.selectTask("classification"),
      action: {
        label: s.sample.action,
        run: async (ctx) => {
          try {
            const made = await createSampleDataset("classification");
            ctx.setDatasetPath(made.path);
            return {
              ok: true,
              message: made.existed
                ? s.sample.existed(folderTail(made.path))
                : s.sample.created(folderTail(made.path)),
            };
          } catch (e) {
            return {
              ok: false,
              message: s.sample.failed(e instanceof Error ? e.message : String(e)),
            };
          }
        },
      },
    },
    {
      anchor: "dataset",
      align: "top",
      title: s.dataset.title,
      body: s.dataset.body,
      waitFor: (facts) => facts.datasetPath.trim() !== "",
      waitHint: s.dataset.waiting,
    },
    {
      anchor: "training",
      align: "top",
      title: s.parameters.title,
      body: s.parameters.body,
    },
    {
      anchor: "train",
      title: s.train.title,
      body: s.train.body,
      waitFor: (facts) => facts.trainingStarted,
      waitHint: s.train.waiting,
    },
    {
      anchor: "training-sheet",
      floating: true,
      title: s.watching.title,
      body: s.watching.body,
    },
    {
      anchor: "training-sheet",
      floating: true,
      title: s.result.title,
      body: s.result.body,
      waitFor: (facts) => facts.trainingEnded,
      waitHint: s.result.waiting,
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
      title: s.others.title,
      body: s.others.body,
    },
  ];
}

export const firstTrainingGuide: GuideDefinition = {
  id: "firstTraining",
  title: (t) => t.guides.firstTraining.title,
  summary: (t) => t.guides.firstTraining.summary,
  steps: firstTrainingSteps,
};
