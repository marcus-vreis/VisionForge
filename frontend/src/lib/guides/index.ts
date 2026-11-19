/** The guides the interface offers, in the order the menu lists them (ADR-115). */

import { firstTrainingGuide } from "./first-training";
import { interfaceTour } from "./interface-tour";
import { ROUTE, routeOpen, type Session } from "../session";
import type { GuideDefinition, GuideId } from "./types";

export const GUIDES: readonly GuideDefinition[] = [
  interfaceTour,
  firstTrainingGuide,
];

/** The guides this server can play: the first training creates the sample
 *  dataset, which a shared server may not do yet (ADR-216). */
export function offeredGuides(session: Session): readonly GuideDefinition[] {
  return routeOpen(session, ROUTE.sampleDataset)
    ? GUIDES
    : GUIDES.filter((guide) => guide.id !== "firstTraining");
}

const BY_ID: Record<GuideId, GuideDefinition> = {
  tour: interfaceTour,
  firstTraining: firstTrainingGuide,
};

export function guideById(id: GuideId): GuideDefinition {
  return BY_ID[id];
}

export { GUIDE_TASKS, guideTaskFor, type GuideTask } from "./tasks";
export type {
  ActionResult,
  GuideAction,
  GuideContext,
  GuideDefinition,
  GuideFacts,
  GuideId,
  GuideStep,
} from "./types";
