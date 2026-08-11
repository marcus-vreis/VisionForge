import { useEffect, useState } from "react";
import { fetchSystemInfo } from "../api/client";

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
export function Header({
  userName,
  onChangeName,
}: { userName?: string; onChangeName?: () => void } = {}) {
  const [time, setTime] = useState(() => new Date());
  // Read from the backend rather than hardcoded: a screenshot of a bug then
  // carries the version that produced it, and the two can never drift.
  const [version, setVersion] = useState("");

  useEffect(() => {
    const id = setInterval(() => setTime(new Date()), 30_000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    fetchSystemInfo()
      .then((info) => setVersion(info.version))
      .catch(() => setVersion(""));
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
            local ai studio{version ? ` · v${version}` : ""}
          </div>
        </div>
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
        {userName && (
          <button
            type="button"
            onClick={onChangeName}
            title="Trocar nome"
            style={{
              display: "flex",
              alignItems: "center",
              gap: 9,
              padding: "9px 14px",
              background: "rgba(255,255,255,0.025)",
              border: "1px solid var(--vf-panel-stroke)",
              borderRadius: 10,
              cursor: "pointer",
              animation: "fadeUp 700ms ease both",
            }}
          >
            <span
              style={{
                width: 6,
                height: 6,
                borderRadius: "50%",
                background: "var(--accent-vf)",
                boxShadow: "0 0 8px var(--accent-glow)",
                animation: "pulse-dot 2.6s ease-in-out infinite",
              }}
            />
            <span
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: 11,
                letterSpacing: "0.12em",
                textTransform: "uppercase",
                color: "var(--vf-text-dim)",
              }}
            >
              Bem-vindo,
            </span>
            {/* Sem text-transform: o nome aparece exatamente como foi digitado. */}
            <span
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: 11,
                letterSpacing: "0.12em",
                color: "var(--vf-text)",
              }}
            >
              {userName}
            </span>
          </button>
        )}
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
      </div>
    </header>
  );
}
