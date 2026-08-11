/** The welcome line, and when it is worth showing.
 *
 * A greeting is the one piece of UI with no job to do, which makes it the
 * easiest to get wrong: shown too often it becomes furniture the eye edits out,
 * and shown with a bad name it is worse than silence.
 */

/** Tidy an OS account name into something worth greeting.
 *
 * Account names carry shapes nobody wants read back at them — a domain prefix,
 * a dotted first.last, an email. What survives is the first token, capitalised.
 * A name that is only digits or symbols is dropped: "Bem-vindo, 12345" reads
 * like a database record, not a hello.
 */
export function displayName(raw: string | null | undefined): string {
  if (!raw) return "";
  const afterDomain = raw.includes("\\") ? raw.slice(raw.lastIndexOf("\\") + 1) : raw;
  const first = afterDomain.split(/[@._\-\s]+/).find((p) => /\p{L}/u.test(p));
  if (!first) return "";
  return first.charAt(0).toLocaleUpperCase("pt-BR") + first.slice(1);
}

/** `Bem-vindo, Marcus` — or a plain welcome when there is no usable name. */
export function greetingFor(raw: string | null | undefined): string {
  const name = displayName(raw);
  return name ? `Bem-vindo, ${name}` : "Bem-vindo";
}

const SEEN_KEY = "vf.greeted";

/** Whether to play the greeting now.
 *
 * Once per session, not once per page load: a reload mid-work is not an
 * arrival, and replaying the animation every time would turn it into a
 * flicker the researcher waits out.
 */
export function shouldGreet(storage: Storage | undefined = globalThis.sessionStorage): boolean {
  if (!storage) return true; // no storage is not a reason to stay silent
  try {
    if (storage.getItem(SEEN_KEY)) return false;
    storage.setItem(SEEN_KEY, "1");
    return true;
  } catch {
    return true;
  }
}
