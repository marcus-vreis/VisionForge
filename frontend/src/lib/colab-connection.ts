/** What the Colab corner of the header shows, decided from the server's status (ADR-163). */

import type { ColabDatasetSource, ColabStatus } from "../api/client";

export type ColabStage =
  | "not_installed"
  | "installing"
  | "install_failed"
  | "logged_out"
  | "login_pending"
  | "connected";

export function colabStage(status: ColabStatus): ColabStage {
  if (status.install_state === "installing") return "installing";
  if (status.installed_version !== status.pinned_version) {
    return status.install_state === "failed" ? "install_failed" : "not_installed";
  }
  if (status.logged_in) return "connected";
  return status.login_pending ? "login_pending" : "logged_out";
}

/** A pasted code is one token: Google's codes have no spaces. */
export function cleanAuthCode(raw: string): string | null {
  const code = raw.trim();
  if (!code || code.length > 512 || /\s/.test(code)) return null;
  return code;
}

/** Read what the person typed as the dataset's source on the VM.
 *
 * `https://…` is a zip; `torchvision:cifar10` and `hf:owner/name` (or
 * `huggingface:`) are public datasets the VM downloads itself. Anything else is
 * null, so the button stays off rather than paying for a VM that will fail.
 */
export function parseDatasetSource(raw: string): ColabDatasetSource | null {
  const text = raw.trim();
  if (/^https?:\/\/\S+$/.test(text)) return { kind: "url", url: text };
  const match = /^(torchvision|huggingface|hf):(\S+)$/.exec(text);
  if (!match) return null;
  const provider = match[1] === "torchvision" ? "torchvision" : "huggingface";
  return { kind: "provider", provider, dataset: match[2] };
}

/** Hours of compute left, rounded so the number reads as an estimate. */
export function roundHours(hours: number): number {
  return hours >= 10 ? Math.round(hours) : Math.round(hours * 10) / 10;
}
