import type { Dict } from "../i18n/pt";
import { useT } from "../i18n/useT";

/** A strategy as the "when to use" sentences name it. */
export type WhenToUseKey = keyof Dict["whenToUse"]["strategies"];

/** One plain sentence under a strategy selector saying when that strategy is
 *  the right one. The selector's own hint says what the strategy does, in the
 *  terms of someone who already knows it; this line is for choosing. */
export function WhenToUse({ strategy }: { strategy: WhenToUseKey }) {
  const t = useT();
  return (
    <p
      data-when-to-use={strategy}
      style={{
        margin: 0,
        fontSize: 13,
        lineHeight: 1.5,
        color: "var(--vf-text-dim)",
      }}
    >
      <span
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: 11,
          letterSpacing: "0.14em",
          textTransform: "uppercase",
          color: "var(--accent-vf)",
          marginRight: 8,
        }}
      >
        {t.whenToUse.label}
      </span>
      {t.whenToUse.strategies[strategy]}
    </p>
  );
}
