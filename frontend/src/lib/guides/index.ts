/** The guides the interface offers, in the order the menu lists them (ADR-115). */

import { firstTrainingGuide } from "./first-training";
import { interfaceTour } from "./interface-tour";
import type { GuideDefinition, GuideId } from "./types";

export const GUIDES: readonly GuideDefinition[] = [
  interfaceTour,
  firstTrainingGuide,
];

const BY_ID: Record<GuideId, GuideDefinition> = {
  tour: interfaceTour,
  firstTraining: firstTrainingGuide,
};

export function guideById(id: GuideId): GuideDefinition {
  return BY_ID[id];
}

export type {
  ActionResult,
  GuideAction,
  GuideContext,
  GuideDefinition,
  GuideFacts,
  GuideId,
  GuideStep,
} from "./types";
