/** What a K-fold, sweep or replicates card will run, in place of a second run
 *  button: the bottom bar's button is the one that starts every strategy, as it
 *  does for classification. When that button is pressed and the card cannot run,
 *  the reason turns into a warning, so the click never fails in silence. */
export function RunSummary({
  text,
  ready,
  reason,
  attempted,
  accent,
}: {
  text: string;
  ready: boolean;
  /** Why the card cannot run; shown muted, and as a warning after a press. */
  reason: string | null;
  attempted: boolean;
  accent: string;
}) {
  const warn = !ready && attempted;
  return (
    <div
      role={warn ? "alert" : undefined}
      style={{
        alignSelf: "center",
        marginLeft: "auto",
        padding: "8px 12px",
        border: `1px solid ${ready ? accent : warn ? "oklch(calc(0.75 - var(--lt-flip, 0) * 0.2) 0.16 60)" : "var(--vf-panel-stroke)"}`,
        borderRadius: 10,
        fontFamily: "var(--font-mono)",
        fontSize: 11.5,
        color: ready ? accent : warn ? "oklch(calc(0.85 - var(--lt-flip, 0) * 0.4) 0.14 70)" : "var(--vf-text-muted)",
        whiteSpace: "nowrap",
      }}
    >
      {ready ? text : (reason ?? text)}
    </div>
  );
}
