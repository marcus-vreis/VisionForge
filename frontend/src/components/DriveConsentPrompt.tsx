/** "Allow Drive access" on the training sheet of a Colab job (ADR-164).
 *
 * The button opens Google's consent page in a new tab. When the window comes
 * back (focus, or the tab visible again) the server is told, and it sends the
 * Enter the Colab CLI waits for: no terminal. "I already allowed" does the same
 * by hand, for a browser whose focus events never fire.
 */
import { useCallback, useEffect, useRef, useState, type CSSProperties } from "react";
import { continueDriveConsent } from "../api/client";
import { consentAnswerer, type DriveConsentPrompt as Prompt } from "../lib/drive-consent";
import { useT } from "../i18n/useT";

const TEXT: CSSProperties = {
  fontFamily: "var(--font-mono)",
  fontSize: 11,
  lineHeight: 1.5,
  color: "var(--vf-text)",
};

const BUTTON: CSSProperties = {
  ...TEXT,
  padding: "6px 10px",
  borderRadius: 8,
  border: "1px solid var(--accent-vf)",
  background: "rgb(var(--lt-ink, 255 255 255) / 0.04)",
  cursor: "pointer",
};

type Phase = "idle" | "sending" | "sent" | "error";

export interface DriveConsentPromptProps {
  prompt: Prompt;
  runId: string;
  /** Injected by the tests; the real call otherwise. */
  send?: (runId: string) => Promise<unknown>;
}

export function DriveConsentPrompt({ prompt, runId, send = continueDriveConsent }: DriveConsentPromptProps) {
  const t = useT().trainingOverlay;
  const [phase, setPhase] = useState<Phase>("idle");
  const answerer = useRef<ReturnType<typeof consentAnswerer> | null>(null);

  const answer = useCallback(() => {
    setPhase("sending");
    send(runId).then(
      () => setPhase("sent"),
      () => setPhase("error"),
    );
  }, [runId, send]);

  useEffect(() => {
    const a = consentAnswerer(answer);
    answerer.current = a;
    const visibility = () => (document.visibilityState === "hidden" ? a.left() : a.returned());
    window.addEventListener("blur", a.left);
    window.addEventListener("focus", a.returned);
    document.addEventListener("visibilitychange", visibility);
    return () => {
      window.removeEventListener("blur", a.left);
      window.removeEventListener("focus", a.returned);
      document.removeEventListener("visibilitychange", visibility);
    };
  }, [answer]);

  const open = () => {
    answerer.current?.opened();
    window.open(prompt.url, "_blank", "noopener,noreferrer");
  };

  return (
    <div
      role="alert"
      data-drive-consent
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 6,
        marginTop: 8,
        padding: "8px 10px",
        borderRadius: 8,
        border: "1px solid var(--accent-vf)",
      }}
    >
      <div style={{ ...TEXT, fontWeight: 600 }}>{t.driveConsentTitle}</div>
      <div style={{ ...TEXT, color: "var(--vf-text-muted)" }}>
        {t.driveConsentBody(prompt.attempt, prompt.attempts)}
      </div>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        <button type="button" style={BUTTON} onClick={open} disabled={phase === "sending"}>
          {t.driveConsentOpen}
        </button>
        <button
          type="button"
          style={{ ...BUTTON, border: "1px solid var(--vf-panel-stroke)" }}
          onClick={answer}
          disabled={phase === "sending"}
        >
          {t.driveConsentDone}
        </button>
      </div>
      {(phase === "sending" || phase === "sent") && (
        <div style={{ ...TEXT, color: "var(--vf-text-muted)" }}>{t.driveConsentSending}</div>
      )}
      {phase === "error" && <div style={TEXT}>{t.driveConsentError}</div>}
    </div>
  );
}
