/** A log line per stage of a Colab VM (issue 355).
 *
 * A job on Colab spends most of its time before the first epoch: allocating the
 * machine, installing VisionForge, preparing the dataset. The server sends a
 * `remote` event at each stage, stamped with `elapsed_s` since the job began,
 * so a sheet opened mid-run still shows when each one happened.
 */

import type { Dict } from "../i18n/pt";
import type { TrainingEvent } from "../types/run";
import type { LogLine } from "./training-log";

type Remote = Extract<TrainingEvent, { event: "remote" }>;
type Stages = Dict["trainingOverlay"]["remoteStages"];

const pad = (n: number): string => String(n).padStart(2, "0");

/** "[m:ss]", or "[h:mm:ss]" past the first hour; "" without a number. */
export function elapsedTag(seconds: unknown): string {
  if (typeof seconds !== "number" || !Number.isFinite(seconds) || seconds < 0) return "";
  const whole = Math.floor(seconds);
  const h = Math.floor(whole / 3600);
  const m = Math.floor((whole % 3600) / 60);
  const s = whole % 60;
  return h > 0 ? `[${h}:${pad(m)}:${pad(s)}]` : `[${m}:${pad(s)}]`;
}

const str = (v: unknown): string => (v === undefined || v === null ? "?" : String(v));

function stageText(s: Stages, e: Remote): string | null {
  switch (e.stage) {
    case "allocated":
      return s.allocated(str(e.accelerator));
    case "kernel_ready":
      return s.kernelReady;
    case "drive_mount":
      return s.driveMount;
    case "drive_consent_sent":
      return s.driveConsentSent;
    case "drive_mounted":
      return s.driveMounted;
    case "drive_copy":
      return s.driveCopy(str(e.files), str(e.seconds));
    case "retry":
      return s.retry;
    case "install":
      return s.install(str(e.version));
    case "dataset":
      return s.dataset;
    case "num_classes":
      return s.numClasses(str(e.old), str(e.new));
    case "train":
      return s.train(str(e.device));
    case "pack":
      return s.pack;
    case "drive_export":
      return e.error ? s.driveExportFailed(str(e.error)) : s.driveExport;
    case "download":
      return s.download;
    case "released":
      return s.released;
    default:
      return null;
  }
}

/** The log line of a VM stage, in the dictionary's language; null for other events. */
export function remoteLogLine(event: TrainingEvent): LogLine | null {
  if (event.event !== "remote") return null;
  const tag = elapsedTag(event.elapsed_s);
  return (d) => {
    const body = stageText(d.trainingOverlay.remoteStages, event) ?? `colab: ${event.stage}`;
    return `> ${tag ? `${tag} ` : ""}${body}`;
  };
}
