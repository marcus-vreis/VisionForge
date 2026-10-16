/** The line under a "Browse" button that says what the native dialog did when it
 *  came back with no path (dismissed, could not open, request failed). */
export function PickerNote({ text }: { text: string | null }) {
  if (!text) return null;
  return (
    <div
      role="status"
      style={{
        gridColumn: "1 / -1",
        marginTop: -6,
        fontFamily: "var(--font-mono)",
        fontSize: 11,
        color: "var(--vf-text-dim)",
      }}
    >
      {text}
    </div>
  );
}
