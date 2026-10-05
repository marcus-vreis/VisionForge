/** Inline marks in a dictionary sentence, parsed into segments.
 *
 * A translation is one whole sentence, so it can move the marked words wherever
 * its grammar wants them; the marks are:
 *
 *   `code`      -> code
 *   **strong**  -> strong
 *   __em__      -> em
 *
 * No nesting: whatever sits inside a mark is shown as written. Identifiers
 * (`n_trials`, `log_uniform`, file names) go inside backticks, which is also
 * what keeps their underscores from reading as emphasis. A delimiter with no
 * partner, or an empty pair such as `` or ****, is plain text.
 */

export type RichKind = "text" | "code" | "strong" | "em";

export interface RichSegment {
  kind: RichKind;
  text: string;
}

// One capture group: String.split puts the matched marks at the odd indices and
// the plain text between them at the even ones, so a plain segment that merely
// starts with a delimiter is never mistaken for a mark.
const MARKS = /(`[^`]+`|\*\*[^*]+\*\*|__[^_]+__)/;

export function parseRich(text: string): RichSegment[] {
  const segments: RichSegment[] = [];
  text.split(MARKS).forEach((part, i) => {
    if (i % 2 === 0) {
      if (part) segments.push({ kind: "text", text: part });
    } else if (part.startsWith("`")) {
      segments.push({ kind: "code", text: part.slice(1, -1) });
    } else if (part.startsWith("**")) {
      segments.push({ kind: "strong", text: part.slice(2, -2) });
    } else {
      segments.push({ kind: "em", text: part.slice(2, -2) });
    }
  });
  return segments;
}
