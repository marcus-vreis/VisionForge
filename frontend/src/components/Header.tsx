import { useEffect, useState } from "react";
import { fetchSystemInfo } from "../api/client";
import { useI18n } from "../i18n/useT";
import type { GuideId } from "../lib/guides";
import { GuideMenu } from "./GuideMenu";

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
  onGuide,
  profileName,
  onChangeProfile,
}: {
  userName?: string;
  onChangeName?: () => void;
  /** Abre um guia escolhido no menu (ADR-104, ADR-115). */
  onGuide?: (id: GuideId) => void;
  /** Perfil em uso (ADR-114). Só vem num servidor que tem perfis: sem ele o chip
   *  não aparece e uma instalação de uma pessoa não vê nada novo. */
  profileName?: string;
  onChangeProfile?: () => void;
} = {}) {
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

  const { t, lang, locale, setLang } = useI18n();

  const dateStr = time.toLocaleDateString(locale, {
    day: "2-digit",
    month: "short",
  });
  const timeStr = time.toLocaleTimeString(locale, {
    hour: "2-digit",
    minute: "2-digit",
  });

  return (
    <header
      style={{
        position: "relative",
        zIndex: 4,
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
        {/* Two real buttons rather than one that flips: each language is a
            target of its own, and aria-pressed says which one is on. */}
        <div
          role="group"
          aria-label={t.language.label}
          style={{
            display: "flex",
            alignItems: "center",
            padding: "0 5px",
            background: "rgba(255,255,255,0.025)",
            border: "1px solid var(--vf-panel-stroke)",
            borderRadius: 10,
            animation: "fadeUp 700ms ease both",
          }}
        >
          {(["pt", "en"] as const).map((code) => (
            <button
              key={code}
              type="button"
              lang={code}
              aria-label={`${code.toUpperCase()} — ${t.language.names[code]}`}
              aria-pressed={code === lang}
              onClick={() => setLang(code)}
              title={code === lang ? undefined : t.language.switchTo}
              style={{
                padding: "9px 6px",
                background: "none",
                border: "none",
                fontFamily: "var(--font-mono)",
                fontSize: 11,
                letterSpacing: "0.12em",
                color: code === lang ? "var(--accent-vf)" : "var(--vf-text-muted)",
                fontWeight: code === lang ? 600 : 400,
                cursor: "pointer",
              }}
            >
              {code.toUpperCase()}
            </button>
          ))}
        </div>
        {onGuide && <GuideMenu onSelect={onGuide} />}
        {profileName && (
          <button
            type="button"
            onClick={onChangeProfile}
            title={t.profile.chipTitle}
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
            <span style={{ fontSize: 12, lineHeight: 1, color: "var(--accent-vf)" }}>
              ▤
            </span>
            <span
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: 11,
                letterSpacing: "0.12em",
                textTransform: "uppercase",
                color: "var(--vf-text-dim)",
              }}
            >
              {t.profile.chip}
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
              {profileName}
            </span>
          </button>
        )}
        {userName && (
          <button
            type="button"
            onClick={onChangeName}
            title={t.header.changeName}
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
              {t.header.welcome}
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
