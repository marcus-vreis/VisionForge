/** "Update available" in the corner of the header, and the button that installs it
 *  (ADR-120).
 *
 * Asks the server once, after the page has drawn. Nothing shows when there is no
 * newer release, when the machine is offline, or when this version was skipped.
 * The button asks for a confirmation, starts the update, and then waits for the
 * restarted server to answer with the new version before reloading the page.
 */

import { useEffect, useState, type CSSProperties } from "react";
import {
  applyUpdate,
  fetchServerVersion,
  fetchUpdateStatus,
  type UpdateStatus,
} from "../api/client";
import { useT } from "../i18n/useT";
import {
  noticeMode,
  readSkipped,
  releaseUrl,
  waitForVersion,
  writeSkipped,
} from "../lib/update-notice";

const MONO: CSSProperties = {
  fontFamily: "var(--font-mono)",
  fontSize: 11,
  letterSpacing: "0.08em",
};

const LINK: CSSProperties = {
  ...MONO,
  background: "none",
  border: "none",
  padding: 0,
  color: "var(--vf-text-dim)",
  textDecoration: "underline",
  cursor: "pointer",
};

type Phase = "idle" | "confirm" | "updating" | "timeout" | "error";

export function UpdateNotice() {
  const t = useT();
  const u = t.update;
  const [status, setStatus] = useState<UpdateStatus | null>(null);
  const [skipped, setSkipped] = useState<string | null>(() => readSkipped());
  const [open, setOpen] = useState(false);
  const [phase, setPhase] = useState<Phase>("idle");
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    let alive = true;
    fetchUpdateStatus()
      .then((s) => alive && setStatus(s))
      .catch(() => alive && setStatus(null));
    return () => {
      alive = false;
    };
  }, []);

  const mode = noticeMode(status, skipped);
  if (mode.kind === "hidden") return null;

  const version = mode.kind === "failed" ? mode.target : mode.latest;

  const start = async () => {
    setPhase("updating");
    setError(null);
    try {
      await applyUpdate();
    } catch (e) {
      setPhase("error");
      setError(e instanceof Error ? e.message : String(e));
      return;
    }
    const outcome = await waitForVersion(version, fetchServerVersion);
    if (outcome === "updated") {
      window.location.reload();
      return;
    }
    if (outcome === "back_old") {
      // The updater recorded the failure; the notice now says so.
      setPhase("idle");
      setOpen(true);
      fetchUpdateStatus().then(setStatus).catch(() => undefined);
      return;
    }
    setPhase("timeout");
  };

  const copy = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // No clipboard permission: the command stays on screen to copy by hand.
    }
  };

  const failed = mode.kind === "failed";
  // After a failure the button comes back when the server can still install:
  // whatever locked a file may have gone.
  const canInstall = mode.kind === "offer" || (failed && status?.can_apply === true);

  return (
    <div style={{ position: "relative", marginTop: 6 }} data-update-notice={mode.kind}>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        style={{
          ...MONO,
          display: "inline-flex",
          alignItems: "center",
          gap: 6,
          padding: "3px 9px",
          borderRadius: 999,
          background: "rgba(255,255,255,0.03)",
          border: `1px solid ${failed ? "oklch(0.7 0.16 25)" : "var(--accent-vf)"}`,
          color: "var(--vf-text)",
          cursor: "pointer",
        }}
      >
        <span
          aria-hidden
          style={{
            width: 6,
            height: 6,
            borderRadius: "50%",
            background: failed ? "oklch(0.7 0.16 25)" : "var(--accent-vf)",
          }}
        />
        {phase === "updating" ? u.updating(version) : failed ? u.failed(version) : u.available(version)}
      </button>

      {open && phase !== "updating" && (
        <div
          role="dialog"
          aria-label={u.available(version)}
          style={{
            position: "absolute",
            top: "calc(100% + 8px)",
            left: 0,
            zIndex: 20,
            width: 360,
            padding: 14,
            borderRadius: 12,
            // Opaque, like the guide menu: the panel sits over the tab bar.
            background: "rgba(10,12,16,0.97)",
            border: "1px solid var(--vf-panel-stroke)",
            boxShadow: "0 12px 30px rgba(0,0,0,0.5)",
            display: "flex",
            flexDirection: "column",
            gap: 10,
          }}
        >
          {mode.kind === "failed" && (
            <div style={{ ...MONO, lineHeight: 1.5, color: "var(--vf-text)" }}>
              {u.failed(mode.target)} {u.logAt(mode.log)}
            </div>
          )}

          {mode.kind === "manual" && (
            <>
              <div style={{ ...MONO, lineHeight: 1.5, color: "var(--vf-text)" }}>
                {u.blocked[mode.reason]}
              </div>
              <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                <code
                  style={{
                    ...MONO,
                    flex: 1,
                    padding: "6px 8px",
                    borderRadius: 6,
                    background: "rgba(0,0,0,0.35)",
                    color: "var(--vf-text)",
                    overflowX: "auto",
                    whiteSpace: "nowrap",
                  }}
                >
                  {mode.command}
                </code>
                <button type="button" style={LINK} onClick={() => copy(mode.command)}>
                  {copied ? u.copied : u.copy}
                </button>
              </div>
            </>
          )}

          {canInstall && phase !== "confirm" && (
            <button
              type="button"
              onClick={() => setPhase("confirm")}
              style={{
                ...MONO,
                padding: "8px 12px",
                borderRadius: 8,
                border: "1px solid var(--accent-vf)",
                background: "rgba(255,255,255,0.04)",
                color: "var(--vf-text)",
                cursor: "pointer",
                textTransform: "uppercase",
              }}
            >
              {failed ? u.retry : u.apply}
            </button>
          )}

          {canInstall && phase === "confirm" && (
            <>
              <div style={{ ...MONO, lineHeight: 1.5, color: "var(--vf-text)" }}>
                {u.confirm(version)}
              </div>
              <div style={{ display: "flex", gap: 10 }}>
                <button
                  type="button"
                  onClick={start}
                  style={{
                    ...MONO,
                    padding: "8px 12px",
                    borderRadius: 8,
                    border: "1px solid var(--accent-vf)",
                    background: "rgba(255,255,255,0.06)",
                    color: "var(--vf-text)",
                    cursor: "pointer",
                    textTransform: "uppercase",
                  }}
                >
                  {u.confirmYes}
                </button>
                <button type="button" style={LINK} onClick={() => setPhase("idle")}>
                  {u.confirmNo}
                </button>
              </div>
            </>
          )}

          {phase === "error" && error && (
            <div style={{ ...MONO, lineHeight: 1.5, color: "oklch(0.75 0.15 25)" }}>{error}</div>
          )}
          {phase === "timeout" && (
            <div style={{ ...MONO, lineHeight: 1.5, color: "var(--vf-text)" }}>{u.timeout}</div>
          )}

          <div style={{ display: "flex", gap: 14 }}>
            <a
              href={releaseUrl(version)}
              target="_blank"
              rel="noreferrer"
              style={{ ...LINK, color: "var(--vf-text-dim)" }}
            >
              {u.whatsNew}
            </a>
            {!failed && (
              <button
                type="button"
                style={LINK}
                onClick={() => {
                  writeSkipped(version);
                  setSkipped(version);
                }}
              >
                {u.skip}
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
