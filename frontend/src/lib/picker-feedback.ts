/** What a native file or folder dialog tells the researcher when it comes back
 *  with no path.
 *
 * The three pick endpoints (routes.py) answer `cancelled: true` both when the
 * dialog was simply dismissed and when it could not open at all (tkinter
 * missing, a container with no display, a Tk error); the flag does not tell the
 * two apart, only the message does. A plain dismissal always carries the same
 * Portuguese sentence, so that one is replaced by the dictionary's text in the
 * language of the interface, and any other message (a real failure, which has
 * something to say) is shown as the server wrote it.
 */

/** The message the server sends when the dialog is dismissed with no choice. */
export const SERVER_CANCEL_MESSAGE = "Cancelado.";

export function pickerCancelText(
  res: { message: string | null },
  canceled: string,
): string {
  if (res.message === null || res.message === SERVER_CANCEL_MESSAGE) return canceled;
  return res.message;
}

/** Run a native picker and hand back the sentence the form should show under it.
 *
 * A chosen path goes to `onPath` and clears the notice (null). A dismissal or a
 * dialog that could not open gets the text of `pickerCancelText`; a request that
 * itself failed gets the error's message, or `failed` when it carries none. The
 * forms without a feedback line used to swallow all three, so a picker that
 * never opened looked like a button that did nothing.
 */
export async function runPicker(
  pick: () => Promise<{ cancelled: boolean; path: string | null; message: string | null }>,
  onPath: (path: string) => void,
  texts: { canceled: string; failed: string },
): Promise<string | null> {
  try {
    const res = await pick();
    if (!res.cancelled && res.path) {
      onPath(res.path);
      return null;
    }
    return pickerCancelText(res, texts.canceled);
  } catch (e) {
    return e instanceof Error && e.message ? e.message : texts.failed;
  }
}
