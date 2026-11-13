/** The Colab corner of the header (ADR-163, phase 1).
 *
 * Installs the pinned CLI, signs this profile in to a Google account with the
 * copy-and-paste flow, shows the account and its balance, and switches account.
 * Nothing here allocates a Colab runtime. The token stays on the server: the
 * page only ever sees the authorization URL and sends back the code.
 *
 * The popover is split in sections (connection, account and usage, train,
 * runtimes), one short line per control; anything longer sits behind a
 * closed "how it works" / "learn more" disclosure.
 */

import { useCallback, useEffect, useState, type CSSProperties, type ReactNode } from "react";
import {
  cancelColabLogin,
  fetchColabStatus,
  fetchColabUsage,
  finishColabLogin,
  installColabCli,
  logoutColab,
  fetchColabSessions,
  startColabLogin,
  stopColabSession,
  type ColabAccelerator,
  type ColabLaunch,
  type ColabSessionInfo,
  type ColabStatus,
  type ColabUsage,
} from "../api/client";
import { useT } from "../i18n/useT";
import {
  cleanAuthCode,
  colabStage,
  parseDatasetSource,
  roundHours,
} from "../lib/colab-connection";
import { loadMyDriveRoot, saveMyDriveRoot } from "../lib/drive-consent";
import { driveLaunchSource } from "../lib/drive-folders";
import { DrivePicker } from "./ColabDrivePicker";
import { MenuSelect } from "./controls/MenuSelect";
import { switchStyle } from "./trackerStyles";
import { ACTION, BODY, CHECK_ROW, DIM, ERROR, FIELD, LABEL, LINK, MONO, STACK } from "./colabStyles";

const INSTALL_POLL_MS = 2000;
const ACCELERATORS: ColabAccelerator[] = ["T4", "L4", "A100", "H100", "G4"];
const isAccelerator = (value: string): value is ColabAccelerator =>
  (ACCELERATORS as string[]).includes(value);

const SECTION_TITLE: CSSProperties = {
  ...LABEL,
  margin: 0,
  fontWeight: 600,
  color: "var(--vf-text-dim)",
};

/** One block of the popover, with a short title and a rule above it. */
function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section
      aria-label={title}
      style={{ ...STACK, gap: 10, paddingTop: 14, borderTop: "1px solid var(--vf-panel-stroke)" }}
    >
      <h3 style={SECTION_TITLE}>{title}</h3>
      {children}
    </section>
  );
}

/** Longer copy, closed until asked for. */
function More({ summary, children }: { summary: string; children: ReactNode }) {
  return (
    <details>
      <summary style={{ ...LINK, display: "list-item", textDecoration: "none" }}>{summary}</summary>
      <p style={{ ...DIM, paddingTop: 6 }}>{children}</p>
    </details>
  );
}

export interface ColabPanelProps {
  status: ColabStatus;
  usage: ColabUsage | null;
  usageError: boolean;
  loginUrl: string | null;
  error: string | null;
  busy: boolean;
  onInstall: () => void;
  onLogin: () => void;
  onFinish: (code: string) => void;
  onCancel: () => void;
  onSwitch: () => void;
  /** Runtimes alive on the account; null until asked. */
  sessions: ColabSessionInfo[] | null;
  onStopSession: (name: string) => void;
  /** Present only when the open task can train on Colab (classification). */
  onTrain?: (launch: ColabLaunch) => void;
  /** Why the open experiment cannot train on Colab now: the button is off. */
  trainBlocked?: string;
}

function TrainOnColab({ busy, onTrain, trainBlocked }: ColabPanelProps) {
  const t = useT();
  const c = t.colab;
  const [source, setSource] = useState("");
  const [accelerator, setAccelerator] = useState<ColabAccelerator>("T4");
  const [bad, setBad] = useState(false);
  const [useDrive, setUseDrive] = useState(false);
  const [driveRoot, setDriveRoot] = useState(loadMyDriveRoot);
  const [drivePath, setDrivePath] = useState("");
  const [driveRemote, setDriveRemote] = useState<string | null>(null);
  const [driveExport, setDriveExport] = useState(false);
  if (!onTrain) return null;

  const submit = () => {
    if (useDrive) {
      saveMyDriveRoot(driveRoot);
      setBad(false);
      onTrain({
        source: driveLaunchSource(driveRoot, drivePath),
        accelerator,
        drive_export: driveExport,
      });
      return;
    }
    const parsed = parseDatasetSource(source);
    setBad(parsed === null);
    if (parsed !== null) onTrain({ source: parsed, accelerator, drive_export: driveExport });
  };

  return (
    <div style={{ ...STACK, gap: 12 }} data-colab-train>
      <label style={CHECK_ROW}>
        <input
          type="checkbox"
          checked={useDrive}
          onChange={(e) => setUseDrive(e.target.checked)}
          style={switchStyle(useDrive)}
        />
        {c.driveSource}
      </label>
      {useDrive ? (
        <>
          <DrivePicker
            driveRoot={driveRoot}
            onDriveRoot={setDriveRoot}
            folder={drivePath}
            remote={driveRemote}
            onPick={(folder, remote) => {
              setDrivePath(folder);
              setDriveRemote(remote);
            }}
          />
          <More summary={c.learnMore}>{c.driveNote}</More>
        </>
      ) : (
        <div style={{ ...STACK, gap: 6 }}>
          <label style={LABEL} htmlFor="colab-source">
            {c.trainSource}
          </label>
          <input
            id="colab-source"
            type="text"
            autoComplete="off"
            spellCheck={false}
            value={source}
            placeholder={c.sourcePlaceholder}
            onChange={(e) => setSource(e.target.value)}
            style={FIELD}
          />
          {bad && <p style={ERROR}>{c.badSource}</p>}
        </div>
      )}
      <label style={CHECK_ROW}>
        <input
          type="checkbox"
          checked={driveExport}
          onChange={(e) => setDriveExport(e.target.checked)}
          style={switchStyle(driveExport)}
        />
        {c.driveExport}
      </label>
      <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
        <span id="colab-accelerator-label" style={LABEL}>
          {c.accelerator}
        </span>
        <div
          role="group"
          aria-labelledby="colab-accelerator-label"
          style={{ display: "flex", flex: 1, minWidth: 0 }}
        >
          <MenuSelect
            size="field"
            value={accelerator}
            title={c.accelerator}
            options={ACCELERATORS}
            onChange={(value) => {
              if (isAccelerator(value)) setAccelerator(value);
            }}
          />
        </div>
        <button
          type="button"
          style={{ ...ACTION, padding: "12px 16px", borderRadius: 10 }}
          disabled={busy || trainBlocked !== undefined}
          onClick={submit}
        >
          {c.train}
        </button>
      </div>
      {trainBlocked !== undefined && (
        <p style={DIM} data-colab-train-blocked>
          {trainBlocked}
        </p>
      )}
      <More summary={c.learnMore}>{c.trainNote}</More>
    </div>
  );
}

function Sessions({ sessions, busy, onStopSession }: ColabPanelProps) {
  const t = useT();
  const c = t.colab;
  if (sessions === null) return null;
  return (
    <div style={{ ...STACK, gap: 6 }} data-colab-sessions>
      {sessions.length === 0 && <p style={DIM}>{c.sessionsNone}</p>}
      {sessions.map((s) => (
        <div
          key={`${s.name}-${s.hardware}`}
          style={{ display: "flex", gap: 12, alignItems: "center", justifyContent: "space-between" }}
        >
          <span style={{ ...MONO, fontSize: 12, color: "var(--vf-text)" }}>
            {s.name} · {s.hardware}
          </span>
          {s.ours ? (
            <button type="button" style={LINK} disabled={busy} onClick={() => onStopSession(s.name)}>
              {c.stopSession}
            </button>
          ) : (
            <span style={{ ...DIM, textAlign: "right" }}>{c.sessionNotOurs}</span>
          )}
        </div>
      ))}
    </div>
  );
}

/** The panel body, without fetching: what each stage of the connection shows. */
export function ColabPanel(props: ColabPanelProps) {
  const t = useT();
  const c = t.colab;
  const { status } = props;
  const stage = colabStage(status);

  return (
    <div style={{ ...STACK, gap: 14 }} data-colab-stage={stage}>
      <header style={{ ...STACK, gap: 4 }}>
        <h2 style={{ ...BODY, fontSize: 15, fontWeight: 600 }}>{c.dialogLabel}</h2>
        <p style={DIM}>{c.intro}</p>
        <More summary={c.howItWorks}>{c.introMore}</More>
      </header>

      {stage !== "connected" && (
        <Section title={c.sectionConnect}>
          {stage === "installing" && <p style={BODY}>{c.installing}</p>}

          {(stage === "not_installed" || stage === "install_failed") && (
            <>
              {stage === "install_failed" && status.install_error && (
                <p style={ERROR}>
                  {c.installFailed} {status.install_error}
                </p>
              )}
              <button
                type="button"
                style={{ ...ACTION, alignSelf: "flex-start" }}
                disabled={props.busy}
                onClick={props.onInstall}
              >
                {stage === "install_failed" ? c.retry : c.install(status.pinned_version)}
              </button>
            </>
          )}

          {(stage === "logged_out" || stage === "login_pending") && <LoginSteps {...props} />}
        </Section>
      )}

      {stage === "connected" && (
        <>
          <Section title={c.sectionAccount}>
            <Connected {...props} />
          </Section>
          {props.onTrain && (
            <Section title={c.trainTitle}>
              <TrainOnColab {...props} />
            </Section>
          )}
          {props.sessions !== null && (
            <Section title={c.sessionsTitle}>
              <Sessions {...props} />
            </Section>
          )}
        </>
      )}

      {props.error && (
        <p role="alert" style={ERROR}>
          {props.error}
        </p>
      )}
    </div>
  );
}

function LoginSteps({ loginUrl, busy, onLogin, onFinish, onCancel }: ColabPanelProps) {
  const t = useT();
  const c = t.colab;
  const [code, setCode] = useState("");
  const [bad, setBad] = useState(false);

  if (!loginUrl) {
    return (
      <>
        <p style={DIM}>{c.perProfile}</p>
        <button type="button" style={{ ...ACTION, alignSelf: "flex-start" }} disabled={busy} onClick={onLogin}>
          {c.login}
        </button>
      </>
    );
  }

  const submit = () => {
    const clean = cleanAuthCode(code);
    setBad(clean === null);
    if (clean !== null) onFinish(clean);
  };

  return (
    <>
      <p style={BODY}>{c.loginStep1}</p>
      <a
        href={loginUrl}
        target="_blank"
        rel="noreferrer"
        style={{ ...LINK, color: "var(--accent-vf)", alignSelf: "flex-start" }}
      >
        {c.openPage}
      </a>
      <label style={BODY} htmlFor="colab-auth-code">
        {c.loginStep2}
      </label>
      <input
        id="colab-auth-code"
        type="text"
        autoComplete="off"
        spellCheck={false}
        value={code}
        placeholder={c.codePlaceholder}
        onChange={(e) => setCode(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && submit()}
        style={FIELD}
      />
      {bad && <p style={ERROR}>{c.badCode}</p>}
      <div style={{ display: "flex", gap: 14, alignItems: "center" }}>
        <button type="button" style={ACTION} disabled={busy} onClick={submit}>
          {c.confirm}
        </button>
        <button type="button" style={LINK} onClick={onCancel}>
          {c.cancel}
        </button>
      </div>
    </>
  );
}

function Connected({ status, usage, usageError, busy, onSwitch }: ColabPanelProps) {
  const t = useT();
  const c = t.colab;
  const units = (n: number) => n.toFixed(2);
  return (
    <>
      <p style={BODY}>{status.account ? c.account(status.account) : c.accountUnknown}</p>
      {usage?.balance != null && <p style={BODY}>{c.balance(units(usage.balance))}</p>}
      {usage?.hours_left != null && usage.rate_per_hour != null && (
        <p style={DIM}>{c.hoursLeft(roundHours(usage.hours_left), units(usage.rate_per_hour))}</p>
      )}
      {usage && usage.hours_left == null && usage.balance != null && <p style={DIM}>{c.noRate}</p>}
      {usageError && <p style={ERROR}>{c.usageFailed}</p>}
      <button type="button" style={{ ...LINK, alignSelf: "flex-start" }} disabled={busy} onClick={onSwitch}>
        {c.switchAccount}
      </button>
    </>
  );
}


const message = (e: unknown) => (e instanceof Error ? e.message : String(e));

/** The header chip and its panel; one instance per profile (keyed by the header). */
export function ColabConnection({
  onTrain,
  trainBlocked,
}: {
  onTrain?: (launch: ColabLaunch) => void;
  trainBlocked?: string;
} = {}) {
  const t = useT();
  const [status, setStatus] = useState<ColabStatus | null>(null);
  const [usage, setUsage] = useState<ColabUsage | null>(null);
  const [usageError, setUsageError] = useState(false);
  const [loginUrl, setLoginUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [open, setOpen] = useState(false);
  const [sessions, setSessions] = useState<ColabSessionInfo[] | null>(null);

  const refresh = useCallback(async () => {
    try {
      setStatus(await fetchColabStatus());
    } catch (e) {
      setError(message(e));
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  // The install runs on the server in the background; ask again until it ends.
  useEffect(() => {
    if (status?.install_state !== "installing") return;
    const id = setInterval(() => void refresh(), INSTALL_POLL_MS);
    return () => clearInterval(id);
  }, [status?.install_state, refresh]);

  // A login left waiting by an earlier page has no URL here: offer a fresh one,
  // which kills the old process on the server.
  const shown = status && { ...status, login_pending: status.login_pending && loginUrl !== null };
  const stage = shown ? colabStage(shown) : null;

  useEffect(() => {
    if (!open || stage !== "connected") return;
    fetchColabUsage()
      .then((u) => {
        setUsage(u);
        setUsageError(false);
      })
      .catch(() => setUsageError(true));
  }, [open, stage]);

  // Asked once connected, not only when the panel opens: a job cut short by a
  // crash may have left a VM running, and the chip says so on load.
  useEffect(() => {
    if (stage !== "connected") return;
    fetchColabSessions()
      .then(setSessions)
      .catch(() => setSessions(null));
  }, [stage]);

  const act = async (step: () => Promise<ColabStatus | void>) => {
    setBusy(true);
    setError(null);
    try {
      const next = await step();
      if (next) setStatus(next);
    } catch (e) {
      setError(message(e));
    } finally {
      setBusy(false);
    }
  };

  const panel: ColabPanelProps | null = shown && {
    status: shown,
    usage,
    usageError,
    loginUrl,
    error,
    busy,
    onInstall: () => void act(installColabCli),
    onLogin: () =>
      void act(async () => {
        setLoginUrl((await startColabLogin()).url);
        return fetchColabStatus();
      }),
    onFinish: (code) =>
      void act(async () => {
        const next = await finishColabLogin(code);
        setLoginUrl(null);
        return next;
      }),
    onCancel: () =>
      void act(async () => {
        setLoginUrl(null);
        return cancelColabLogin();
      }),
    onSwitch: () =>
      void act(async () => {
        setUsage(null);
        setSessions(null);
        return logoutColab();
      }),
    sessions,
    onStopSession: (name) =>
      void act(async () => {
        setSessions(await stopColabSession(name));
      }),
    onTrain: onTrain
      ? (launch) => {
          setOpen(false);
          onTrain(launch);
        }
      : undefined,
    trainBlocked,
  };
  const leftovers = sessions?.filter((s) => s.ours).length ?? 0;

  return (
    <div style={{ position: "relative" }} data-colab-connection={stage ?? "loading"}>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        style={{
          ...MONO,
          display: "inline-flex",
          alignItems: "center",
          gap: 6,
          padding: "6px 10px",
          borderRadius: 999,
          background: "rgb(var(--lt-ink, 255 255 255) / 0.03)",
          border: `1px solid ${stage === "connected" ? "var(--accent-vf)" : "var(--vf-panel-stroke)"}`,
          color: stage === "connected" ? "var(--vf-text)" : "var(--vf-text-muted)",
          cursor: "pointer",
        }}
      >
        <span
          aria-hidden
          style={{
            width: 6,
            height: 6,
            borderRadius: "50%",
            background: stage === "connected" ? "var(--accent-vf)" : "var(--vf-text-muted)",
          }}
        />
        {stage ? t.colab.chip[stage] : t.colab.chip.not_installed}
        {leftovers > 0 && ` · ${t.colab.chipRuntimes(leftovers)}`}
      </button>

      {open && (
        <div
          role="dialog"
          aria-label={t.colab.dialogLabel}
          style={{
            position: "absolute",
            top: "calc(100% + 8px)",
            right: 0,
            zIndex: 20,
            // Wide enough for a Drive path, never wider than the window; tall
            // content scrolls inside the panel instead of running off-screen.
            width: "min(440px, calc(100vw - 32px))",
            maxHeight: "calc(100vh - 96px)",
            overflowY: "auto",
            overscrollBehavior: "contain",
            boxSizing: "border-box",
            padding: 18,
            borderRadius: 12,
            // Opaque, like the update notice: the panel sits over the tab bar.
            background: "rgb(var(--lt-surface, 10 12 16) / 0.97)",
            border: "1px solid var(--vf-panel-stroke)",
            boxShadow: "0 12px 30px rgb(var(--lt-shade, 0 0 0) / 0.5)",
          }}
        >
          {panel ? <ColabPanel {...panel} /> : <p style={ERROR}>{error ?? t.colab.loadFailed}</p>}
        </div>
      )}
    </div>
  );
}
