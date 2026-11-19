import { useEffect, useId, useRef, useState, type CSSProperties } from "react";

import { useT } from "../i18n/useT";
import { greeting } from "../lib/greeting";
import type { ProfileInfo, Pronoun } from "../lib/profile";
import { ROUTE, routeOpen, useSession } from "../lib/session";
import { ProfileTrackingSettings } from "./ProfileTrackingSettings";

interface ProfileMenuProps {
  /** The name on the chip: the profile's, as typed. */
  name: string;
  pronoun?: Pronoun | null;
  /** The profile in use; without it (still loading) the menu has no integrations. */
  profile: ProfileInfo | null;
  /** More than one profile: offer "switch" instead of "new". */
  shared: boolean;
  onProfileSaved: (profile: ProfileInfo) => void;
  onChangeName: () => void;
  onSwitchProfile: () => void;
  onNewProfile: () => void;
  /** For the tests, which render it without a DOM to click. */
  initialOpen?: boolean;
}

const ACTION: CSSProperties = {
  display: "block",
  width: "100%",
  padding: "9px 10px",
  background: "transparent",
  border: "none",
  borderRadius: 8,
  textAlign: "left",
  fontSize: 14,
  color: "var(--vf-text)",
  cursor: "pointer",
};

const SECTION: CSSProperties = {
  fontFamily: "var(--font-mono)",
  fontSize: 12,
  letterSpacing: "0.12em",
  textTransform: "uppercase",
  color: "var(--vf-text-muted)",
  marginBottom: 12,
};

/** The header's profile chip and its menu: integrations, name and profile.
 *
 * One chip, present once there is a name, on a one-profile install too. The
 * menu is a non-modal dialog (it holds form fields, so not a `menu`): it takes
 * focus when it opens, and closes on Esc (focus back on the chip), on a click
 * outside it, and when focus leaves it.
 */
export function ProfileMenu({
  name,
  pronoun,
  profile,
  shared,
  onProfileSaved,
  onChangeName,
  onSwitchProfile,
  onNewProfile,
  initialOpen = false,
}: ProfileMenuProps) {
  const t = useT();
  // On a shared server the profile is the account the login proxy named: there
  // is none to switch to or create (ADR-216); its integrations are the
  // person's own (ADR-223).
  const session = useSession();
  const canEditProfile = routeOpen(session, ROUTE.profileUpdate);
  const canChooseProfile = routeOpen(session, ROUTE.profileCreate);
  const [open, setOpen] = useState(initialOpen);
  const root = useRef<HTMLDivElement>(null);
  const chip = useRef<HTMLButtonElement>(null);
  const panel = useRef<HTMLDivElement>(null);
  const panelId = useId();

  useEffect(() => {
    if (!open) return;
    panel.current?.focus();
    const onDown = (e: MouseEvent) => {
      if (root.current && !root.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      setOpen(false);
      chip.current?.focus();
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const act = (run: () => void) => () => {
    setOpen(false);
    run();
  };

  return (
    <div
      ref={root}
      style={{ position: "relative" }}
      onBlur={(e) => {
        // Tabbing past the last field leaves the menu: it closes behind it.
        const next = e.relatedTarget as Node | null;
        if (open && next && !e.currentTarget.contains(next)) setOpen(false);
      }}
    >
      <button
        ref={chip}
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-controls={open ? panelId : undefined}
        title={t.profile.menuTitle}
        style={{
          display: "flex",
          alignItems: "center",
          gap: 9,
          padding: "9px 14px",
          background: open ? "var(--accent-soft)" : "rgb(var(--lt-ink, 255 255 255) / 0.025)",
          border: `1px solid ${open ? "var(--accent-vf)" : "var(--vf-panel-stroke)"}`,
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
        {/* The greeting gives way first on a narrow header (index.css): the
            name is what identifies the profile. */}
        <span
          className="vf-profile-greeting"
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: 11,
            letterSpacing: "0.12em",
            textTransform: "uppercase",
            color: "var(--vf-text-dim)",
          }}
        >
          {`${greeting(t, pronoun)},`}
        </span>
        {/* Sem text-transform: o nome aparece exatamente como foi digitado. */}
        <span
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: 11,
            letterSpacing: "0.12em",
            color: "var(--vf-text)",
            maxWidth: 220,
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}
        >
          {name}
        </span>
      </button>

      {open && (
        <div
          ref={panel}
          id={panelId}
          role="dialog"
          aria-label={t.profile.menuTitle}
          tabIndex={-1}
          style={{
            position: "absolute",
            top: "calc(100% + 8px)",
            right: 0,
            width: "min(360px, calc(100vw - 32px))",
            maxHeight: "min(72vh, 620px)",
            overflowY: "auto",
            padding: 16,
            background: "rgb(var(--lt-surface, 10 12 16) / 0.97)",
            border: "1px solid var(--vf-panel-stroke)",
            borderRadius: 12,
            boxShadow: "0 24px 70px rgb(var(--lt-shade, 0 0 0) / 0.6)",
            outline: "none",
            zIndex: 10,
          }}
        >
          <div
            style={{
              fontFamily: "var(--font-display)",
              fontSize: 17,
              fontWeight: 600,
              color: "var(--vf-text)",
              marginBottom: 16,
              overflowWrap: "anywhere",
            }}
          >
            {name}
          </div>

          {session.server_mode && (
            <div
              data-server-account
              style={{
                marginTop: -8,
                marginBottom: 16,
                fontSize: 13,
                lineHeight: 1.5,
                color: "var(--vf-text-dim)",
              }}
            >
              {session.user && (
                <div
                  style={{
                    fontFamily: "var(--font-mono)",
                    fontSize: 12,
                    color: "var(--vf-text)",
                    overflowWrap: "anywhere",
                  }}
                >
                  {t.server.account(session.user)}
                </div>
              )}
              {session.profile && (
                <div
                  style={{
                    fontFamily: "var(--font-mono)",
                    fontSize: 12,
                    overflowWrap: "anywhere",
                  }}
                >
                  {t.profile.folder(session.profile)}
                </div>
              )}
              <div style={{ marginTop: 6 }}>{t.server.menuNote}</div>
            </div>
          )}

          {profile && canEditProfile && (
            <section aria-label={t.profile.integrations}>
              <div style={SECTION}>
                {t.profile.integrations}
                <span style={{ textTransform: "none", letterSpacing: 0, marginLeft: 8 }}>
                  · {t.profile.integrationsNote}
                </span>
              </div>
              <ProfileTrackingSettings profile={profile} onSaved={onProfileSaved} />
            </section>
          )}

          <div
            style={{
              marginTop: profile && canEditProfile ? 16 : 0,
              paddingTop: profile && canEditProfile ? 10 : 0,
              borderTop:
                profile && canEditProfile ? "1px solid var(--vf-panel-stroke)" : "none",
              display: "flex",
              flexDirection: "column",
              gap: 2,
            }}
          >
            <button type="button" style={ACTION} onClick={act(onChangeName)}>
              {t.header.changeName}
            </button>
            {canChooseProfile &&
              (shared ? (
                <button type="button" style={ACTION} onClick={act(onSwitchProfile)}>
                  {t.profile.switchProfile}
                </button>
              ) : (
                <button type="button" style={ACTION} onClick={act(onNewProfile)}>
                  {t.profile.createTitle}
                </button>
              ))}
          </div>
        </div>
      )}
    </div>
  );
}
