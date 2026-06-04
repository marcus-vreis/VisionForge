import { useMemo } from "react";

/** Build a smooth sine path using quadratic bezier control points. */
function buildWave(
  width: number,
  amp: number,
  period: number,
  y: number,
  phase = 0,
): string {
  const segments = Math.ceil(width / (period / 2)) + 2;
  let d = `M ${-period} ${y}`;
  for (let i = 0; i < segments; i++) {
    const x0 = -period + i * (period / 2);
    const x1 = x0 + period / 4;
    const x2 = x0 + period / 2;
    const dir = (i + phase) % 2 === 0 ? -1 : 1;
    d += ` Q ${x1} ${y + amp * dir} ${x2} ${y}`;
  }
  return d;
}

interface WaveLayerProps {
  count: number;
  baseY: number;
  spread: number;
  amp: number;
  period: number;
  opacity: number;
  blur?: number;
  strokeW: number;
}

function WaveLayer({
  count,
  baseY,
  spread,
  amp,
  period,
  opacity,
  blur = 0,
  strokeW,
}: WaveLayerProps) {
  const lines = useMemo(() => {
    const out: { d: string; key: number }[] = [];
    for (let i = 0; i < count; i++) {
      const y = baseY + (i - count / 2) * (spread / count);
      const phase = i % 2;
      const a = amp + (i % 3) * 6;
      const p = period + (i % 4) * 40;
      out.push({ d: buildWave(3000, a, p, y, phase), key: i });
    }
    return out;
  }, [count, baseY, spread, amp, period]);

  return (
    <g
      style={{
        // Static, not animated: a continuously drifting full-viewport layer
        // forces every backdrop-filter panel above it to re-blur each frame,
        // which collapsed the whole UI to ~5fps. The waves stay as a frozen
        // backdrop instead. See `prefers-reduced-motion` note in index.css.
        filter: blur ? `blur(${blur}px)` : "none",
        opacity,
      }}
    >
      {lines.map((l) => (
        <path
          key={l.key}
          d={l.d}
          fill="none"
          stroke="var(--accent-vf)"
          strokeWidth={strokeW}
          strokeLinecap="round"
        />
      ))}
    </g>
  );
}

/** Animated SVG wave background. Color follows --accent-vf. */
export function Waves() {
  return (
    <svg
      width="100%"
      height="100%"
      viewBox="0 0 1600 900"
      preserveAspectRatio="xMidYMid slice"
      style={{
        position: "absolute",
        inset: 0,
        width: "100%",
        height: "100%",
        pointerEvents: "none",
        zIndex: 0,
        // No mix-blend-mode: blending a full-viewport SVG into the page forces
        // every backdrop-filter panel above it onto a slow offscreen-buffer
        // compositing path (the residual jank after the wave animation was
        // frozen). On the near-black background, plain accent strokes read the
        // same as `screen` did. See the WaveLayer comment + index.css.
      }}
    >
      <defs>
        <linearGradient id="vfFadeMask" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stopColor="white" stopOpacity="0" />
          <stop offset="30%" stopColor="white" stopOpacity="1" />
          <stop offset="70%" stopColor="white" stopOpacity="1" />
          <stop offset="100%" stopColor="white" stopOpacity="0" />
        </linearGradient>
        <mask id="vfFade">
          <rect width="1600" height="900" fill="url(#vfFadeMask)" />
        </mask>
      </defs>
      <g mask="url(#vfFade)">
        <WaveLayer
          count={6}
          baseY={450}
          spread={520}
          amp={80}
          period={360}
          opacity={0.18}
          blur={2}
          strokeW={1.2}
        />
        <WaveLayer
          count={9}
          baseY={450}
          spread={580}
          amp={48}
          period={280}
          opacity={0.36}
          blur={0.4}
          strokeW={0.9}
        />
        <WaveLayer
          count={14}
          baseY={450}
          spread={620}
          amp={28}
          period={220}
          opacity={0.55}
          strokeW={0.55}
        />
      </g>
    </svg>
  );
}

/** Floating particle dots that pulse with the current accent color. */
export function Particles() {
  const dots = useMemo(() => {
    const out: { x: number; y: number; s: number; d: number; delay: number }[] =
      [];
    let seed = 42;
    const rng = () => {
      seed = (seed * 9301 + 49297) % 233280;
      return seed / 233280;
    };
    for (let i = 0; i < 40; i++) {
      out.push({
        x: rng() * 100,
        y: rng() * 100,
        s: 1 + rng() * 2.5,
        d: 6 + rng() * 10,
        delay: rng() * -10,
      });
    }
    return out;
  }, []);

  return (
    <div
      style={{ position: "absolute", inset: 0, pointerEvents: "none", zIndex: 0 }}
    >
      {dots.map((d, i) => (
        <span
          key={i}
          style={{
            position: "absolute",
            left: `${d.x}%`,
            top: `${d.y}%`,
            width: d.s,
            height: d.s,
            background: "var(--accent-vf)",
            borderRadius: "999px",
            boxShadow: "0 0 6px var(--accent-glow)",
            opacity: 0.7,
          }}
        />
      ))}
    </div>
  );
}
