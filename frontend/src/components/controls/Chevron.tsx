/** The disclosure arrow, as geometry rather than as a character.
 *
 * It used to be the glyph `▾` inside an inline-block span with
 * `transform: rotate(180deg)`. A glyph does not sit in the middle of its own
 * box — the line box carries ascender and descender space above and below it —
 * so rotating the box swung the triangle through an arc instead of turning it
 * in place. The result read as a wobble, most visible on the architecture
 * picker where the arrow is next to text that does not move.
 *
 * An SVG whose viewBox is centred on the triangle rotates around its own
 * centre, because now the box and the shape share one.
 */
export function Chevron({
  open,
  size = 10,
  color = "var(--vf-text-muted)",
}: {
  open: boolean;
  size?: number;
  color?: string;
}) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="-6 -6 12 12"
      aria-hidden="true"
      focusable="false"
      style={{
        display: "block",
        flexShrink: 0,
        color,
        transform: open ? "rotate(180deg)" : "rotate(0deg)",
        transformOrigin: "50% 50%",
        transition: "transform 200ms ease",
      }}
    >
      {/* Centred on (0,0): the arrow spans -3..3 vertically, so its own middle
          is the origin the rotation turns about. */}
      <path
        d="M -4 -2 L 0 2.6 L 4 -2"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
