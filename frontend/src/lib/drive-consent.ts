/** The Google Drive consent of a Colab job, read from its events (ADR-164).
 *
 * Every new Colab runtime asks for it once: the job sends
 * `drive_consent_required` with Google's page, the person allows it in a new tab
 * and the GUI tells the server when they come back. A 400 makes the server ask
 * again with a new page, so only the latest request counts, and it is answered
 * once the job says it sent the reply, mounted the Drive, or ended.
 */
import type { TrainingEvent } from "../types/run";

export interface DriveConsentPrompt {
  url: string;
  attempt: number;
  attempts: number;
}

const ANSWERED = new Set(["drive_consent_sent", "drive_mounted"]);

export function driveConsentPrompt(events: TrainingEvent[]): DriveConsentPrompt | null {
  for (let i = events.length - 1; i >= 0; i -= 1) {
    const e = events[i];
    if (e.event === "drive_consent_required") {
      return { url: e.url, attempt: e.attempt, attempts: e.attempts };
    }
    if (e.event === "end" || (e.event === "remote" && ANSWERED.has(e.stage))) return null;
  }
  return null;
}

/** When the window's focus says the person came back from the consent page.
 *
 * Only a return counts: the window must first have lost the focus (or been
 * hidden) after the page was opened, otherwise a click inside the app would
 * answer before Google was even shown.
 */
export class ConsentReturn {
  private opened = false;
  private left = false;

  open(): void {
    this.opened = true;
    this.left = false;
  }

  leave(): void {
    if (this.opened) this.left = true;
  }

  /** True once per trip away and back. */
  back(): boolean {
    if (!this.opened || !this.left) return false;
    this.opened = false;
    this.left = false;
    return true;
  }
}

const ROOT_KEY = "vf.colab.myDriveRoot";

/** The "My Drive" folder the person gave by hand, remembered in this browser. */
export function loadMyDriveRoot(): string {
  try {
    return window.localStorage.getItem(ROOT_KEY) ?? "";
  } catch {
    return "";
  }
}

export function saveMyDriveRoot(root: string): void {
  try {
    if (root.trim()) window.localStorage.setItem(ROOT_KEY, root.trim());
    else window.localStorage.removeItem(ROOT_KEY);
  } catch {
    // Private window or blocked storage: the field just starts empty next time.
  }
}

/** What the consent card does with the window's events and its buttons.
 *
 * `opened` when "Allow Drive access" opens Google's page; `left` and
 * `returned` from blur/focus and visibilitychange; `allowed` from "I already
 * allowed", the fallback for a browser whose focus never comes back.
 */
export function consentAnswerer(send: () => void) {
  const trip = new ConsentReturn();
  return {
    opened: () => trip.open(),
    left: () => trip.leave(),
    returned: () => {
      if (trip.back()) send();
    },
    allowed: () => send(),
  };
}
