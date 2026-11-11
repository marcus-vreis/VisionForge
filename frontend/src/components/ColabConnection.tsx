/** The Colab corner of the header (ADR-163, phase 1).
 *
 * Installs the pinned CLI, signs this profile in to a Google account with the
 * copy-and-paste flow, shows the account and its balance, and switches account.
 * Nothing here allocates a Colab runtime. The token stays on the server: the
 * page only ever sees the authorization URL and sends back the code.
 */

import { useCallback, useEffect, useState, type CSSProperties } from "react";
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

const MONO: CSSProperties = {
  fontFamily: "var(--font-mono)",
  fontSize: 11,
  letterSpacing: "0.08em",
};

const TEXT: CSSProperties = { ...MONO, lineHeight: 1.5, color: "var(--vf-text)" };
const ERROR: CSSProperties = {
  ...TEXT,
  color: "oklch(calc(0.75 - var(--lt-flip, 0) * 0.2) 0.15 25)",
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

const ACTION: CSSProperties = {
  ...MONO,
  padding: "8px 12px",
  borderRadius: 8,
  border: "1px solid var(--accent-vf)",
  background: "rgb(var(--lt-ink, 255 255 255) / 0.04)",
  color: "var(--vf-text)",
  cursor: "pointer",
  textTransform: "uppercase",
};

const INSTALL_POLL_MS = 2000;
const ACCELERATORS: ColabAccelerator[] = ["T4", "L4", "A100", "H100", "G4"];

const FIELD: CSSProperties = {
  ...MONO,
  padding: "6px 8px",
  borderRadius: 6,
  border: "1px solid var(--vf-panel-stroke)",
  background: "rgb(var(--lt-shade, 0 0 0) / 0.35)",
  color: "var(--vf-text)",
};

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
  const [driveExport, setDriveExport] = useState(false);
  if (!onTrain) return null;

  const submit = () => {
    if (useDrive) {
      saveMyDriveRoot(driveRoot);
      setBad(false);
      const root = driveRoot.trim();
      onTrain({
        source: root ? { kind: "drive", my_drive_root: root } : { kind: "drive" },
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
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }} data-colab-train>
      <div style={{ ...TEXT, fontWeight: 600 }}>{c.trainTitle}</div>
      <label style={{ ...TEXT, display: "flex", gap: 6, alignItems: "center" }}>
        <input type="checkbox" checked={useDrive} onChange={(e) => setUseDrive(e.target.checked)} />
        {c.driveSource}
      </label>
      {useDrive ? (
        <>
          <label style={TEXT} htmlFor="colab-drive-root">
            {c.driveRoot}
          </label>
          <input
            id="colab-drive-root"
            type="text"
            autoComplete="off"
            spellCheck={false}
            value={driveRoot}
            placeholder={c.driveRootPlaceholder}
            onChange={(e) => setDriveRoot(e.target.value)}
            style={FIELD}
          />
          <div style={{ ...TEXT, color: "var(--vf-text-dim)" }}>{c.driveNote}</div>
        </>
      ) : (
        <>
          <label style={TEXT} htmlFor="colab-source">
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
          {bad && <div style={ERROR}>{c.badSource}</div>}
        </>
      )}
      <label style={{ ...TEXT, display: "flex", gap: 6, alignItems: "center" }}>
        <input
          type="checkbox"
          checked={driveExport}
          onChange={(e) => setDriveExport(e.target.checked)}
        />
        {c.driveExport}
      </label>
      <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
        <label style={TEXT} htmlFor="colab-accelerator">
          {c.accelerator}
        </label>
        <select
          id="colab-accelerator"
          value={accelerator}
          onChange={(e) => setAccelerator(e.target.value as ColabAccelerator)}
          style={FIELD}
        >
          {ACCELERATORS.map((a) => (
            <option key={a} value={a}>
              {a}
            </option>
          ))}
        </select>
        <button
          type="button"
          style={ACTION}
          disabled={busy || trainBlocked !== undefined}
          onClick={submit}
        >
          {c.train}
        </button>
      </div>
      {trainBlocked !== undefined && (
        <div style={TEXT} data-colab-train-blocked>
          {trainBlocked}
        </div>
      )}
      <div style={{ ...TEXT, color: "var(--vf-text-dim)" }}>{c.trainNote}</div>
    </div>
  );
}

function Sessions({ sessions, busy, onStopSession }: ColabPanelProps) {
  const t = useT();
  const c = t.colab;
  if (sessions === null) return null;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }} data-colab-sessions>
      <div style={{ ...TEXT, fontWeight: 600 }}>{c.sessionsTitle}</div>
      {sessions.length === 0 && <div style={{ ...TEXT, color: "var(--vf-text-dim)" }}>{c.sessionsNone}</div>}
      {sessions.map((s) => (
        <div key={`${s.name}-${s.hardware}`} style={{ display: "flex", gap: 10, alignItems: "center" }}>
          <span style={TEXT}>
            {s.name} · {s.hardware}
          </span>
          {s.ours ? (
            <button type="button" style={LINK} disabled={busy} onClick={() => onStopSession(s.name)}>
              {c.stopSession}
            </button>
          ) : (
            <span style={{ ...TEXT, color: "var(--vf-text-dim)" }}>{c.sessionNotOurs}</span>
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
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }} data-colab-stage={stage}>
      <div style={{ ...TEXT, color: "var(--vf-text-dim)" }}>{c.intro}</div>

      {stage === "installing" && <div style={TEXT}>{c.installing}</div>}

      {(stage === "not_installed" || stage === "install_failed") && (
        <>
          {stage === "install_failed" && status.install_error && (
            <div style={ERROR}>
              {c.installFailed} {status.install_error}
            </div>
          )}
          <button type="button" style={ACTION} disabled={props.busy} onClick={props.onInstall}>
            {stage === "install_failed" ? c.retry : c.install(status.pinned_version)}
          </button>
        </>
      )}

      {(stage === "logged_out" || stage === "login_pending") && (
        <LoginSteps {...props} />
      )}

      {stage === "connected" && (
        <>
          <Connected {...props} />
          <Sessions {...props} />
          <TrainOnColab {...props} />
        </>
      )}

      {props.error && <div style={ERROR}>{props.error}</div>}
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
        <div style={TEXT}>{c.perProfile}</div>
        <button type="button" style={ACTION} disabled={busy} onClick={onLogin}>
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
      <div style={TEXT}>{c.loginStep1}</div>
      <a href={loginUrl} target="_blank" rel="noreferrer" style={{ ...LINK, color: "var(--accent-vf)" }}>
        {c.openPage}
      </a>
      <label style={TEXT} htmlFor="colab-auth-code">
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
        style={{
          ...MONO,
          padding: "6px 8px",
          borderRadius: 6,
          border: "1px solid var(--vf-panel-stroke)",
          background: "rgb(var(--lt-shade, 0 0 0) / 0.35)",
          color: "var(--vf-text)",
        }}
      />
      {bad && <div style={ERROR}>{c.badCode}</div>}
      <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
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
      <div style={TEXT}>{status.account ? c.account(status.account) : c.accountUnknown}</div>
      {usage?.balance != null && <div style={TEXT}>{c.balance(units(usage.balance))}</div>}
      {usage?.hours_left != null && usage.rate_per_hour != null && (
        <div style={TEXT}>{c.hoursLeft(roundHours(usage.hours_left), units(usage.rate_per_hour))}</div>
      )}
      {usage && usage.hours_left == null && usage.balance != null && (
        <div style={{ ...TEXT, color: "var(--vf-text-dim)" }}>{c.noRate}</div>
      )}
      {usageError && <div style={ERROR}>{c.usageFailed}</div>}
      <div style={{ display: "flex", gap: 14 }}>
        <button type="button" style={LINK} disabled={busy} onClick={onSwitch}>
          {c.switchAccount}
        </button>
      </div>
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
            width: 360,
            padding: 14,
            borderRadius: 12,
            // Opaque, like the update notice: the panel sits over the tab bar.
            background: "rgb(var(--lt-surface, 10 12 16) / 0.97)",
            border: "1px solid var(--vf-panel-stroke)",
            boxShadow: "0 12px 30px rgb(var(--lt-shade, 0 0 0) / 0.5)",
          }}
        >
          {panel ? <ColabPanel {...panel} /> : <div style={ERROR}>{error ?? t.colab.loadFailed}</div>}
        </div>
      )}
    </div>
  );
}
