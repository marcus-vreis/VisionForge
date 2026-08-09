/** Tell the researcher a run ended, when they are no longer looking.
 *
 * The tool is built to leave an evening of experiments queued and walk away,
 * and until now the only way to learn a run finished was to come back and look.
 *
 * Two channels, deliberately, because they fail differently. A browser
 * notification reaches someone in another window but can be denied, unsupported,
 * or silently swallowed by an OS focus mode. The tab title always works, needs
 * no permission, and is visible from the tab strip — so it is the one that
 * carries the guarantee, and the notification is the bonus on top.
 */

export type RunOutcome = "completed" | "failed";

const BASE_TITLE = "VisionForge — Local AI Training";

/** The tab title for a finished run: readable at tab-strip width, mark first. */
export function titleFor(outcome: RunOutcome, label: string): string {
  const mark = outcome === "completed" ? "✓" : "✗";
  return `${mark} ${label} — VisionForge`;
}

/** What the notification says. Kept to one line: OS toasts truncate the rest. */
export function messageFor(
  outcome: RunOutcome,
  label: string,
  detail?: string,
): { title: string; body: string } {
  if (outcome === "completed") {
    return {
      title: `Treino concluído — ${label}`,
      body: detail ? detail : "Abra o VisionForge para ver os resultados.",
    };
  }
  return {
    title: `Treino falhou — ${label}`,
    body: detail ? detail : "Abra o VisionForge para ver o erro.",
  };
}

/** Ask for permission, once a run is actually under way.
 *
 * Deliberately not called on page load: a permission prompt before the user has
 * done anything is the pattern people reflexively deny, and a denial is sticky.
 * Asking when they start a run ties the prompt to a reason.
 */
export async function requestPermission(
  notification: typeof Notification | undefined = globalThis.Notification,
): Promise<NotificationPermission | "unsupported"> {
  if (!notification) return "unsupported";
  if (notification.permission !== "default") return notification.permission;
  try {
    return await notification.requestPermission();
  } catch {
    return "denied";
  }
}

/** Whether a notification would actually be shown. */
export function canNotify(
  notification: typeof Notification | undefined = globalThis.Notification,
): boolean {
  return Boolean(notification) && notification!.permission === "granted";
}

/** Restore the title once the researcher is looking again. */
export function resetTitle(doc: Document = document): void {
  doc.title = BASE_TITLE;
}

/** Announce a finished run through both channels.
 *
 * The title is set unconditionally; the notification only when it is allowed
 * and the page is hidden — a toast for a window the user is already staring at
 * is noise.
 */
export function announce(
  outcome: RunOutcome,
  label: string,
  detail?: string,
  deps: {
    doc?: Document;
    notification?: typeof Notification;
    hidden?: boolean;
  } = {},
): void {
  const doc = deps.doc ?? document;
  const notification = deps.notification ?? globalThis.Notification;
  const hidden = deps.hidden ?? doc.hidden;

  doc.title = titleFor(outcome, label);
  if (!hidden || !canNotify(notification)) return;

  const { title, body } = messageFor(outcome, label, detail);
  try {
    new notification!(title, { body, tag: "visionforge-run" });
  } catch {
    // The title already carries the message; a failed toast changes nothing.
  }
}
