// Animated wave background. Two big stacks of SVG sine paths drift horizontally
// at different speeds. Color comes from the current --accent CSS var so the
// whole scene shifts color when the task tab changes.

const { useMemo } = React;

// Build a smooth sine path using cubic bezier control points.
// width: total path width, amp: amplitude, period: pixels per full wave, y: center y
function buildWave(width, amp, period, y, phase = 0) {
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

function WaveLayer({ count, baseY, spread, amp, period, duration, opacity, blur, strokeW }) {
  const lines = useMemo(() => {
    const out = [];
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
    <g style={{
      animation: `drift ${duration}s linear infinite`,
      transformOrigin: 'center',
      filter: blur ? `blur(${blur}px)` : 'none',
      opacity,
    }}>
      {lines.map(l => (
        <path
          key={l.key}
          d={l.d}
          fill="none"
          stroke="var(--accent)"
          strokeWidth={strokeW}
          strokeLinecap="round"
        />
      ))}
    </g>
  );
}

function Waves() {
  return (
    <svg
      width="100%"
      height="100%"
      viewBox="0 0 1600 900"
      preserveAspectRatio="xMidYMid slice"
      style={{
        position: 'absolute',
        inset: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
        zIndex: 0,
        mixBlendMode: 'screen',
      }}
    >
      <defs>
        <linearGradient id="fadeMask" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stopColor="white" stopOpacity="0" />
          <stop offset="30%" stopColor="white" stopOpacity="1" />
          <stop offset="70%" stopColor="white" stopOpacity="1" />
          <stop offset="100%" stopColor="white" stopOpacity="0" />
        </linearGradient>
        <mask id="vfFade">
          <rect width="1600" height="900" fill="url(#fadeMask)" />
        </mask>
      </defs>

      <g mask="url(#vfFade)">
        {/* big soft layer in background, blurred */}
        <WaveLayer
          count={6} baseY={450} spread={520}
          amp={80} period={360} duration={28}
          opacity={0.18} blur={2} strokeW={1.2}
        />
        {/* mid drifting layer */}
        <WaveLayer
          count={9} baseY={450} spread={580}
          amp={48} period={280} duration={42}
          opacity={0.36} blur={0.4} strokeW={0.9}
        />
        {/* sharp foreground filaments */}
        <WaveLayer
          count={14} baseY={450} spread={620}
          amp={28} period={220} duration={64}
          opacity={0.55} blur={0} strokeW={0.55}
        />
      </g>
    </svg>
  );
}

// Floating particle dots that match the sketch's tiny dots between the waves.
function Particles() {
  const dots = useMemo(() => {
    const out = [];
    const rng = (seed => () => {
      seed = (seed * 9301 + 49297) % 233280;
      return seed / 233280;
    })(42);
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
    <div style={{ position: 'absolute', inset: 0, pointerEvents: 'none', zIndex: 0 }}>
      {dots.map((d, i) => (
        <span key={i} style={{
          position: 'absolute',
          left: `${d.x}%`,
          top: `${d.y}%`,
          width: d.s,
          height: d.s,
          background: 'var(--accent)',
          borderRadius: '999px',
          boxShadow: '0 0 6px var(--accent-glow)',
          opacity: 0.7,
          animation: `pulse-dot ${d.d}s ease-in-out ${d.delay}s infinite`,
        }} />
      ))}
    </div>
  );
}

window.Waves = Waves;
window.Particles = Particles;
