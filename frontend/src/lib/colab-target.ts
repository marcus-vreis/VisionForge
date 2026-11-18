/** Colab as a place to train, chosen where the GPU is chosen (ADR-211).
 *
 * The device selector offers the account's Colab beside this machine's
 * devices; with it picked, the usual Treinar button sends the classification
 * to `/api/colab/experiment/run`. Where the VM takes the dataset from is asked
 * on the Dados step, since the VM cannot read this computer's folders.
 *
 * Pure on purpose: no DOM and no fetch, so it is covered by plain vitest.
 */

import type {
  ColabAccelerator,
  ColabLaunch,
  ColabSessionInfo,
  ColabUsage,
} from "../api/client";
import { parseDatasetSource } from "./colab-connection";
import { driveLaunchSource } from "./drive-folders";

export const COLAB_ACCELERATORS: readonly ColabAccelerator[] = ["T4", "L4", "A100", "H100", "G4"];

/** Where the VM takes the dataset from, and what the run asks of the account. */
export interface ColabSetup {
  accelerator: ColabAccelerator;
  /** A folder in "My Drive", or a link the VM downloads by itself. */
  kind: "drive" | "link";
  /** A zip URL, `torchvision:<name>` or `hf:<owner/name>`. */
  link: string;
  /** This computer's "My Drive" folder, when the detection needs help. */
  driveRoot: string;
  /** The folder picked in the Drive browser, relative to "My Drive". */
  drivePath: string;
  /** That folder as the VM sees it, to show before the run. */
  driveRemote: string | null;
  /** Also save the run to MyDrive/VisionForge/runs/ (ADR-164). */
  driveExport: boolean;
}

export function defaultColabSetup(driveRoot = ""): ColabSetup {
  return {
    accelerator: "T4",
    kind: "drive",
    link: "",
    driveRoot,
    drivePath: "",
    driveRemote: null,
    driveExport: false,
  };
}

export type ColabDataProblem = "bad_link" | "no_folder";

/**
 * Why the dataset origin cannot be sent yet, or null.
 *
 * A Drive origin without a picked folder falls back to the form's own dataset
 * folder (the server refuses it when it is outside "My Drive"); with neither
 * there is nothing to send.
 */
export function colabDataProblem(setup: ColabSetup, formBaseDir: string): ColabDataProblem | null {
  if (setup.kind === "link") return parseDatasetSource(setup.link) === null ? "bad_link" : null;
  return setup.drivePath.trim() === "" && formBaseDir.trim() === "" ? "no_folder" : null;
}

/** What the run adds to the classification payload, or null while the origin
 *  cannot be sent. */
export function colabLaunchOf(setup: ColabSetup, formBaseDir: string): ColabLaunch | null {
  if (colabDataProblem(setup, formBaseDir) !== null) return null;
  const source =
    setup.kind === "link"
      ? parseDatasetSource(setup.link)
      : driveLaunchSource(setup.driveRoot, setup.drivePath);
  if (source === null) return null;
  return { source, accelerator: setup.accelerator, drive_export: setup.driveExport };
}

export type ColabBlock = "task" | "replicates" | "comparison";

/**
 * Why the open experiment cannot train on Colab, or null.
 *
 * Colab trains one classification `config.block` (ADR-163): not another task,
 * not a researcher's own, and not a replicate set (ADR-056) or a replicated
 * comparison (ADR-193), which are several runs.
 */
export function colabBlock(open: {
  task: string;
  custom: boolean;
  strategy: string;
}): ColabBlock | null {
  if (open.custom || open.task !== "classification") return "task";
  if (open.strategy === "replicates") return "replicates";
  if (open.strategy === "replicatedComparison") return "comparison";
  return null;
}

/** What only Colab has, for the selector's button: the balance in compute
 *  units and how many runtimes are alive on the account. */
export interface ColabAccountFacts {
  balance: string | null;
  runtimes: number;
  /** Hardware of the runtimes alive, e.g. `["T4"]`; empty while none is. */
  allocated: string[];
}

export function colabAccountFacts(
  usage: ColabUsage | null,
  sessions: readonly ColabSessionInfo[] | null,
): ColabAccountFacts {
  const balance =
    typeof usage?.balance === "number" && Number.isFinite(usage.balance)
      ? usage.balance.toFixed(usage.balance >= 100 ? 0 : 1)
      : null;
  const alive = sessions ?? [];
  return {
    balance,
    runtimes: alive.length,
    allocated: [...new Set(alive.map((s) => s.hardware).filter(Boolean))],
  };
}

/**
 * The dataset origin in one line, for the Revisão step: the link, or the Drive
 * folder as the VM sees it when the browser gave that path, else the folder
 * here. Empty while nothing is set.
 */
export function colabOriginLabel(setup: ColabSetup, formBaseDir: string): string {
  if (setup.kind === "link") return setup.link.trim();
  return setup.driveRemote ?? (setup.drivePath.trim() || formBaseDir.trim());
}
