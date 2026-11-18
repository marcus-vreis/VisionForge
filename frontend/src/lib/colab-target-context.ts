import { createContext } from "react";

import type { ColabDataProblem, ColabSetup } from "./colab-target";

/** What the Dados step needs while Colab is the place to train (ADR-211). */
export interface ColabTargetValue {
  setup: ColabSetup;
  onChange: (patch: Partial<ColabSetup>) => void;
  /** The form's own dataset folder, the Drive origin's fallback. */
  formBaseDir: string;
  problem: ColabDataProblem | null;
}

/** Null while the training runs on this machine. */
export const ColabTargetContext = createContext<ColabTargetValue | null>(null);
