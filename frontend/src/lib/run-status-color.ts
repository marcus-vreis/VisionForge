/** Status dot color of a run: completed green, running blue, failed red. Each
 *  lightness follows `--lt-flip`, so the dot reads on the light theme too. */
export function statusColor(status: string): string {
  if (status === "completed") return "oklch(calc(0.78 - var(--lt-flip, 0) * 0.26) 0.18 150)";
  if (status === "running") return "oklch(calc(0.74 - var(--lt-flip, 0) * 0.18) 0.16 240)";
  if (status === "failed") return "oklch(calc(0.74 - var(--lt-flip, 0) * 0.18) 0.18 22)";
  return "var(--vf-text-muted)";
}

/** A run's status in the reader's language; one the dictionary does not know
 *  is shown as the server wrote it. */
export function runStatusLabel(status: string, words: Readonly<Record<string, string>>): string {
  return Object.hasOwn(words, status) ? words[status] : status;
}
