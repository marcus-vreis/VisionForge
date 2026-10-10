/** The first-run tour as a guide (ADR-104, ADR-115): the same seven stops, none gated. */

import { tourSteps } from "../tour";
import type { GuideDefinition } from "./types";

export const interfaceTour: GuideDefinition = {
  id: "tour",
  title: (t) => t.guides.tour.title,
  summary: (t) => t.guides.tour.summary,
  // A tour step has no `waitFor`, `action`, `onEnter` or `floating`, which is
  // what keeps it playing exactly as it did before guides had gates.
  steps: (t) => tourSteps(t),
};
