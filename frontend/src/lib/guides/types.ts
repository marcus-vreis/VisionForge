/** What a guide is (ADR-115).
 *
 * The first-run tour (ADR-104) only points at things; a guide may also wait for
 * the researcher to do something and offer a button that does a step for them.
 * Both are expressed here so one component can play either.
 */

import type { Dict } from "../../i18n/pt";
import type { TourStep } from "../tour";

export type GuideId = "tour" | "firstTraining";

/** What a step may ask about the app. Read from the real screen state, never kept apart. */
export interface GuideFacts {
  /** The classification form's dataset folder, as typed or picked. */
  datasetPath: string;
  /** The training stream produced `start` after the guide opened. */
  trainingStarted: boolean;
  /** The same run then produced `end`. */
  trainingEnded: boolean;
}

/** What a step's action or `onEnter` may change in the app. Supplied by App. */
export interface GuideContext {
  /** Switches to a task tab by key. */
  selectTask: (key: string) => void;
  /** Fills the dataset field of the classification form. */
  setDatasetPath: (path: string) => void;
  /** Puts the training sheet away without stopping the run. */
  hideTrainingSheet: () => void;
}

/** The outcome an action shows on the card. */
export interface ActionResult {
  ok: boolean;
  message: string;
}

export interface GuideAction {
  label: string;
  run: (ctx: GuideContext) => Promise<ActionResult>;
}

export interface GuideStep extends TourStep {
  /** Open when the researcher has done what the step asks. Absent: always open. */
  waitFor?: (facts: GuideFacts) => boolean;
  /** One line shown while `waitFor` is false, saying what is being waited for. */
  waitHint?: string;
  /** A button on the card that does the step. */
  action?: GuideAction;
  /** Runs when the step opens. */
  onEnter?: (ctx: GuideContext) => void;
  /** Scroll the target near the top of the screen instead of the middle, so a tall
   *  card has the room below it rather than covering what it points at. */
  align?: "top";
  /** The card sits beside its target and the page is not dimmed, because the step
   *  is about watching the screen behind it. */
  floating?: boolean;
}

export interface GuideDefinition {
  id: GuideId;
  title: (t: Dict) => string;
  /** One line under the title in the menu and the invitation. */
  summary: (t: Dict) => string;
  steps: (t: Dict) => GuideStep[];
}
