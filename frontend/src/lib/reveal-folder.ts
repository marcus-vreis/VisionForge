/** Opening a run's folder from the History.
 *
 * The window the server opens appears on the *server's* desktop, which is the
 * researcher's own only when the browser and the server share a machine. The
 * server decides that from the client address (`can_reveal` on the run detail)
 * and refuses the request otherwise (403), so the page neither guesses from
 * `window.location` nor offers a button that cannot work.
 *
 * Pure on purpose: no DOM and no fetch, so it is covered by plain vitest.
 */

/** The three sentences a refused or failed request is told in. */
export interface RevealTexts {
  forbidden: string;
  notFound: string;
  failed: string;
}

/** Whether the "open folder" button is shown: only when the server said so.
 *  A payload from an older server has no flag, which reads as "not offered". */
export function canOfferReveal(
  detail: { can_reveal?: boolean } | null | undefined,
): boolean {
  return detail?.can_reveal === true;
}

/** What to tell the researcher when the request did not open the folder. */
export function revealErrorText(status: number, texts: RevealTexts): string {
  if (status === 403) return texts.forbidden;
  if (status === 404) return texts.notFound;
  return texts.failed;
}
