import { useEffect, useRef, useState } from "react";

import {
  normalizeUserName,
  readUserName,
  saveUserName,
} from "../lib/user-name";

type Phase = "boot" | "hello" | "ask" | "exit" | "done";

interface WelcomeOverlayProps {
  /** Chamado quando o nome está definido (primeira visita ou visita seguinte). */
  onName: (name: string) => void;
  /** Força a introdução completa mesmo com nome salvo (usado pelo "trocar nome"). */
  forceAsk?: boolean;
}

/** Introdução de primeira execução (ADR-090).
 *
 * Primeira visita: a tela escurece, "Bem-vindo" entra, sai, e "Qual é o seu
 * nome?" abre uma linha no meio para digitar. O nome é salvo e passa a
 * aparecer no header.
 *
 * Visitas seguintes: só o cumprimento "Bem-vindo, {Nome}" por ~2s e entra —
 * nunca pergunta de novo. Quem quiser trocar clica no chip do header.
 */
export function WelcomeOverlay({ onName, forceAsk = false }: WelcomeOverlayProps) {
  const saved = forceAsk ? "" : readUserName();
  const [phase, setPhase] = useState<Phase>("boot");
  const [name, setName] = useState(saved);
  const [draft, setDraft] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);
  const returning = saved.length > 0;

  useEffect(() => {
    const timers: number[] = [];
    const at = (ms: number, fn: () => void) =>
      timers.push(window.setTimeout(fn, ms));

    if (returning) {
      at(60, () => setPhase("hello"));
      at(2100, () => setPhase("exit"));
      at(2900, () => {
        setPhase("done");
        onName(saved);
      });
    } else {
      at(160, () => setPhase("hello"));
      at(2300, () => setPhase("ask"));
      at(3000, () => inputRef.current?.focus());
    }
    return () => timers.forEach(clearTimeout);
    // Roda uma vez por montagem: o fluxo é uma sequência, não um efeito reativo.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    const value = normalizeUserName(draft);
    if (!value) {
      inputRef.current?.focus();
      return;
    }
    saveUserName(value);
    setName(value);
    setPhase("exit");
    window.setTimeout(() => {
      setPhase("done");
      onName(value);
    }, 850);
  };

  // O overlay some do fluxo depois da saída para não capturar cliques.
  if (phase === "done") return null;

  const visible = phase === "hello" || phase === "ask";
  const helloOn = phase === "hello";
  const askOn = phase === "ask";
  const canEnter = draft.trim().length > 0;

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 40,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: `rgba(4,5,7,${phase === "boot" ? 0 : 0.78})`,
        backdropFilter: `blur(${visible ? 10 : 0}px)`,
        WebkitBackdropFilter: `blur(${visible ? 10 : 0}px)`,
        opacity: phase === "exit" ? 0 : 1,
        transition:
          "background 700ms ease, backdrop-filter 700ms ease, opacity 650ms ease",
      }}
    >
      <div
        style={{
          position: "absolute",
          width: 520,
          height: 520,
          borderRadius: "50%",
          border: "1px solid var(--accent-soft)",
          pointerEvents: "none",
          opacity: visible ? 1 : 0,
          transition: "opacity 800ms ease",
          animation: "vfRing 4.6s ease-out infinite",
        }}
      />

      <div
        style={{
          position: "absolute",
          textAlign: "center",
          pointerEvents: "none",
          opacity: helloOn ? 1 : 0,
          transform: `translateY(${
            phase === "boot" ? "22px" : helloOn ? "0px" : "-30px"
          }) scale(${helloOn ? 1 : 0.97})`,
          filter: `blur(${helloOn ? 0 : 6}px)`,
          transition:
            "opacity 900ms ease, transform 1100ms cubic-bezier(.16,.84,.24,1), filter 900ms ease",
        }}
      >
        <div
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: 11,
            letterSpacing: "0.3em",
            textTransform: "uppercase",
            color: "var(--vf-text-muted)",
            marginBottom: 18,
          }}
        >
          VisionForge
        </div>
        <div
          style={{
            fontFamily: "var(--font-display)",
            fontSize: 74,
            fontWeight: 600,
            letterSpacing: "-0.03em",
            lineHeight: 1,
            color: "var(--vf-text)",
          }}
        >
          {returning ? `Bem-vindo, ${name}` : "Bem-vindo"}
        </div>
      </div>

      <div
        style={{
          position: "absolute",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          opacity: askOn ? 1 : 0,
          transform: `translateY(${
            askOn ? "0px" : phase === "exit" ? "-18px" : "26px"
          })`,
          filter: `blur(${askOn ? 0 : 5}px)`,
          pointerEvents: askOn ? "auto" : "none",
          transition:
            "opacity 850ms ease 120ms, transform 950ms cubic-bezier(.16,.84,.24,1) 120ms, filter 850ms ease",
        }}
      >
        <div
          style={{
            fontFamily: "var(--font-display)",
            fontSize: 38,
            fontWeight: 500,
            letterSpacing: "-0.02em",
            color: "var(--vf-text)",
            textAlign: "center",
          }}
        >
          Qual é o seu nome?
        </div>

        <form
          onSubmit={submit}
          style={{
            marginTop: 38,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: 26,
          }}
        >
          <div
            style={{
              position: "relative",
              width: "min(520px, 78vw)",
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
            }}
          >
            <input
              ref={inputRef}
              type="text"
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              placeholder="digite aqui"
              autoComplete="off"
              spellCheck={false}
              maxLength={40}
              aria-label="Seu nome"
              style={{
                width: "100%",
                textAlign: "center",
                background: "transparent",
                border: "none",
                outline: "none",
                padding: "6px 4px 14px",
                fontFamily: "var(--font-display)",
                fontSize: 30,
                letterSpacing: "-0.01em",
                color: "var(--vf-text)",
                caretColor: "var(--accent-vf)",
              }}
            />
            <div
              style={{
                height: 1,
                width: askOn ? "100%" : "0%",
                background:
                  "linear-gradient(90deg, transparent, var(--accent-vf), transparent)",
                boxShadow: "0 0 14px var(--accent-glow)",
                transition: "width 900ms cubic-bezier(.16,.84,.24,1) 380ms",
              }}
            />
          </div>

          <button
            type="submit"
            style={{
              padding: "12px 30px",
              background: canEnter ? "var(--accent-soft)" : "transparent",
              border: `1px solid ${
                canEnter ? "var(--accent-vf)" : "rgba(255,255,255,0.10)"
              }`,
              borderRadius: 12,
              color: canEnter ? "var(--vf-text)" : "var(--vf-text-muted)",
              fontFamily: "var(--font-mono)",
              fontSize: 12,
              letterSpacing: "0.18em",
              textTransform: "uppercase",
              cursor: "pointer",
              boxShadow: canEnter ? "inset 0 0 18px var(--accent-glow)" : "none",
              transition: "all 400ms ease",
            }}
          >
            Entrar ↵
          </button>
        </form>
      </div>
    </div>
  );
}
