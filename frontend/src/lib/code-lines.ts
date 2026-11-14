import type { CodeNote } from "../api/client";

/** One line of a function, as the code view of "Como funciona" draws it. */
export interface LineMarks {
  /** Notes (by index) whose range starts on this line: their markers go in its gutter. */
  starts: number[];
  /** Notes whose range ends on this line: an open one is shown right under it. */
  ends: number[];
  /** Some note covers the line: the gutter draws a quiet bar along the range. */
  noted: boolean;
  /** Only branches this run does not take cover it: the line is faded. */
  faded: boolean;
}

/** The marks of each of `count` lines, from the notes of their function. A line a
 *  note on this run's path covers is never faded, even when a skipped branch
 *  covers it too. */
export function lineMarks(notes: readonly CodeNote[], count: number): LineMarks[] {
  return Array.from({ length: count }, (_, line) => {
    const covering = notes.filter((n) => n.first <= line && line <= n.last);
    return {
      starts: notes.flatMap((n, i) => (n.first === line ? [i] : [])),
      ends: notes.flatMap((n, i) => (n.last === line ? [i] : [])),
      noted: covering.length > 0,
      faded: covering.some((n) => n.active === false) && !covering.some((n) => n.active === true),
    };
  });
}
