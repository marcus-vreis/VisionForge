import { useState } from "react";

import { advanceLatch, guideFacts, openLatch } from "./gates";
import type { GuideFacts } from "./types";

/** The facts a guide's gates read, from the training events and the dataset field.
 *
 * Meant to live as long as the guide is open: the first render records what the
 * event list already holds (a finished run's events do not count, see
 * `gates.ts`), and later renders fold new events into the latch. The latch is
 * folded during render, the documented way to derive state from props without
 * an effect and the extra render it costs.
 */
export function useGuideFacts(
  events: readonly { event: string }[],
  datasetPath: string,
): GuideFacts {
  const [latch, setLatch] = useState(() => openLatch(events));
  const folded = advanceLatch(latch, events);
  if (folded !== latch) setLatch(folded);
  return guideFacts(folded, datasetPath);
}
