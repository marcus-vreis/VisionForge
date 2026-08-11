import { useEffect, useState } from "react";

import { greetingFor, shouldGreet } from "../lib/greeting";

/** A greeting that arrives, is read, and leaves.
 *
 * Top-right, above everything, and gone on its own after a few seconds. It
 * never waits for a click: a dismissable banner is a decision the researcher
 * has to make about something that asked nothing of them.
 *
 * The name comes from `/api/health`, which reads the OS account — the machine
 * already knows who is sitting at it, so there is no field to fill in. If that
 * request fails the component renders nothing at all rather than greeting a
 * stranger.
 */
export function WelcomeBanner() {
  const [greeting, setGreeting] = useState<string | null>(null);
  const [leaving, setLeaving] = useState(false);

  useEffect(() => {
    if (!shouldGreet()) return;
    let alive = true;
    fetch("/api/health")
      .then((r) => (r.ok ? r.json() : null))
      .then((health: { user?: string } | null) => {
        if (alive) setGreeting(greetingFor(health?.user));
      })
      .catch(() => {
        // A greeting is not worth reporting a failure over.
      });
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    if (greeting === null) return;
    const fade = setTimeout(() => setLeaving(true), 4200);
    const drop = setTimeout(() => setGreeting(null), 5000);
    return () => {
      clearTimeout(fade);
      clearTimeout(drop);
    };
  }, [greeting]);

  if (greeting === null) return null;

  return (
    <>
      <style>{`
        @keyframes vf-welcome-in {
          from { opacity: 0; transform: translateY(-14px) scale(0.97); }
          to   { opacity: 1; transform: translateY(0) scale(1); }
        }
        @keyframes vf-welcome-sheen {
          from { background-position: -180% 0; }
          to   { background-position: 280% 0; }
        }
        /* Someone who set "reduce motion" asked not to be moved; the greeting
           still appears, it just stops sliding and sweeping. */
        @media (prefers-reduced-motion: reduce) {
          .vf-welcome { animation: none !important; }
          .vf-welcome-text { animation: none !important; background: none !important;
                             -webkit-text-fill-color: currentColor !important; }
        }
      `}</style>
      <div
        className="vf-welcome"
        role="status"
        aria-live="polite"
        style={{
          position: "fixed",
          top: 18,
          right: 22,
          zIndex: 60,
          padding: "10px 18px",
          borderRadius: 999,
          background: "rgba(14,16,20,0.72)",
          border: "1px solid var(--vf-panel-stroke)",
          backdropFilter: "blur(14px)",
          fontFamily: "var(--font-mono)",
          fontSize: 12,
          letterSpacing: "0.08em",
          pointerEvents: "none",
          animation: "vf-welcome-in 620ms cubic-bezier(0.16, 1, 0.3, 1)",
          opacity: leaving ? 0 : 1,
          transform: leaving ? "translateY(-10px)" : "none",
          transition: "opacity 760ms ease, transform 760ms ease",
        }}
      >
        <span
          className="vf-welcome-text"
          style={{
            background:
              "linear-gradient(90deg, var(--vf-text-dim) 0%, var(--vf-text) 45%, var(--accent-vf, #6fb3ff) 55%, var(--vf-text-dim) 100%)",
            backgroundSize: "220% 100%",
            WebkitBackgroundClip: "text",
            backgroundClip: "text",
            WebkitTextFillColor: "transparent",
            animation: "vf-welcome-sheen 2400ms ease-out 260ms both",
          }}
        >
          {greeting}
        </span>
      </div>
    </>
  );
}
