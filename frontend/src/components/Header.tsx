import { useEffect, useState } from "react";

function Logo() {
  return (
    <div
      style={{
        width: 40,
        height: 40,
        borderRadius: 10,
        background:
          "radial-gradient(circle at 30% 30%, var(--accent-soft), rgba(8,10,14,0.8))",
        border: "1px solid var(--accent-vf)",
        boxShadow:
          "0 0 20px var(--accent-glow), inset 0 0 14px var(--accent-soft)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        position: "relative",
        flexShrink: 0,
      }}
    >
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
        <circle
          cx="12"
          cy="12"
          r="4"
          stroke="var(--accent-vf)"
          strokeWidth="1.6"
        />
        <circle
          cx="12"
          cy="12"
          r="9"
          stroke="var(--accent-vf)"
          strokeWidth="1"
          strokeDasharray="2 3"
          opacity="0.7"
        />
        <path
          d="M12 3 V6 M12 18 V21 M3 12 H6 M18 12 H21"
          stroke="var(--accent-vf)"
          strokeWidth="1.2"
          strokeLinecap="round"
        />
      </svg>
    </div>
  );
}

/** Top header with logo, brand name, and clock. Device selection lives only in the BottomBar. */
export function Header() {
  const [time, setTime] = useState(() => new Date());

  useEffect(() => {
    const id = setInterval(() => setTime(new Date()), 30_000);
    return () => clearInterval(id);
  }, []);

  const dateStr = time.toLocaleDateString("pt-BR", {
    day: "2-digit",
    month: "short",
  });
  const timeStr = time.toLocaleTimeString("pt-BR", {
    hour: "2-digit",
    minute: "2-digit",
  });

  return (
    <header
      style={{
        position: "relative",
        zIndex: 3,
        padding: "22px 40px 0",
        maxWidth: 1280,
        margin: "0 auto",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
        <Logo />
        <div>
          <div
            style={{
              fontSize: 22,
              fontWeight: 700,
              letterSpacing: "-0.01em",
              lineHeight: 1,
              fontFamily: "var(--font-display)",
              color: "var(--vf-text)",
            }}
          >
            Vision
            <span
              style={{
                color: "var(--accent-vf)",
                textShadow: "0 0 12px var(--accent-glow)",
              }}
            >
              Forge
            </span>
          </div>
          <div
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: 11,
              color: "var(--vf-text-muted)",
              letterSpacing: "0.16em",
              textTransform: "uppercase",
              marginTop: 4,
            }}
          >
            local ai studio · v0.0.1
          </div>
        </div>
      </div>

      <div
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: 11,
          color: "var(--vf-text-muted)",
          letterSpacing: "0.12em",
          textTransform: "uppercase",
        }}
      >
        {dateStr} · {timeStr}
      </div>
    </header>
  );
}
