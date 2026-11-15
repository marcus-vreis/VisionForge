/** The progress bar of a background job (incorporate and merge). */
export function Progress({ pct }: { pct: number }) {
  return (
    <div
      role="progressbar"
      aria-valuenow={pct}
      aria-valuemin={0}
      aria-valuemax={100}
      style={{ height: 6, borderRadius: 3, background: "rgb(var(--lt-ink, 255 255 255) / 0.08)" }}
    >
      <div style={{ height: "100%", width: `${pct}%`, borderRadius: 3, background: "var(--accent-vf)" }} />
    </div>
  );
}
