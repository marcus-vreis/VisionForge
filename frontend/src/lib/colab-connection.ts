/** What the Colab corner of the header shows, decided from the server's status (ADR-163). */

import type { ColabStatus } from "../api/client";

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

/** Hours of compute left, rounded so the number reads as an estimate. */
export function roundHours(hours: number): number {
  return hours >= 10 ? Math.round(hours) : Math.round(hours * 10) / 10;
}
