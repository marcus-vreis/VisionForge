/** Colab as a place to train (ADR-211): its rows in the device selector and
 *  the card of the Dados step that asks where the VM takes the dataset from. */

import { useContext, type CSSProperties } from "react";

import type { ColabAccelerator } from "../api/client";
import type { ColabAccount } from "../hooks/useColabAccount";
import { useT } from "../i18n/useT";
import { parseDatasetSource } from "../lib/colab-connection";
import { COLAB_ACCELERATORS } from "../lib/colab-target";
import { ColabTargetContext } from "../lib/colab-target-context";
import { saveMyDriveRoot } from "../lib/drive-consent";
import { DrivePicker } from "./ColabDrivePicker";
import { Segmented, TextField } from "./controls";
import { CHECK_ROW, DIM, ERROR } from "./colabStyles";
import { switchStyle } from "./trackerStyles";

/** What the device selector is told about Colab by the page. */
export interface ColabDeviceChoice {
  /** Colab is where the next training runs. */
  active: boolean;
  accelerator: ColabAccelerator;
  /** Why the open experiment cannot train there; the options are off. */
  blocked?: string;
  onPick: (accelerator: ColabAccelerator) => void;
}

const MENU_KICKER: CSSProperties = {
  padding: "8px 12px 2px",
  fontFamily: "var(--font-mono)",
  fontSize: 10,
  letterSpacing: "0.16em",
  textTransform: "uppercase",
  color: "var(--vf-text-muted)",
};

const MENU_NOTE: CSSProperties = {
  ...DIM,
  fontSize: 12,
  padding: "0 12px 6px",
};

/** Colab's rows in the device menu: what the account has, then one button per
 *  GPU. Off, with the reason, while the account is not connected or the open
 *  experiment cannot train there. */
export function ColabDeviceSection({
  account,
  choice,
}: {
  account: ColabAccount;
  choice: ColabDeviceChoice;
}) {
  const c = useT().colab.target;
  const connected = account.stage === "connected";
  const reason = !connected ? c.notConnected : choice.blocked;
  const { balance, runtimes, allocated } = account.facts;
  const facts = [
    account.email ? c.account(account.email) : null,
    balance !== null ? c.balance(balance) : null,
    runtimes > 0 ? c.runtimes(runtimes) : null,
    allocated.length > 0 ? c.allocated(allocated.join(", ")) : null,
  ].filter((fact): fact is string => fact !== null);

  return (
    <div
      data-colab-target={account.stage ?? "loading"}
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 6,
        marginTop: 4,
        paddingTop: 4,
        borderTop: "1px solid var(--vf-panel-stroke)",
      }}
    >
      <div style={MENU_KICKER}>{c.menuTitle}</div>
      {facts.length > 0 && <p style={MENU_NOTE}>{facts.join(" · ")}</p>}
      <div
        role="group"
        aria-label={c.optionSubtitle}
        style={{ display: "flex", flexWrap: "wrap", gap: 6, padding: "0 10px 4px" }}
      >
        {COLAB_ACCELERATORS.map((gpu) => {
          const on = choice.active && choice.accelerator === gpu;
          return (
            <button
              key={gpu}
              type="button"
              aria-pressed={on}
              disabled={reason !== undefined}
              title={c.option(gpu)}
              onClick={() => choice.onPick(gpu)}
              style={{
                padding: "8px 9px",
                fontFamily: "var(--font-mono)",
                fontSize: 12,
                fontWeight: 600,
                borderRadius: 10,
                border: "1px solid",
                borderColor: on ? "var(--accent-vf)" : "var(--vf-panel-stroke)",
                background: on ? "var(--accent-soft)" : "transparent",
                color: reason !== undefined ? "var(--vf-text-muted)" : "var(--vf-text)",
                cursor: reason !== undefined ? "not-allowed" : "pointer",
              }}
            >
              {gpu}
            </button>
          );
        })}
      </div>
      {reason !== undefined && (
        <p style={MENU_NOTE} data-colab-target-off>
          {reason}
        </p>
      )}
    </div>
  );
}

const CARD: CSSProperties = {
  background: "var(--vf-panel)",
  border: "1px solid var(--accent-vf)",
  borderRadius: 18,
  padding: 26,
  minWidth: 0,
  display: "grid",
  gap: 14,
};

const KICKER: CSSProperties = {
  fontFamily: "var(--font-mono)",
  fontSize: 10,
  color: "var(--vf-text-muted)",
  letterSpacing: "0.20em",
  textTransform: "uppercase",
};

/** The Dados step's card while Colab is the place to train: where the VM takes
 *  the dataset from. Draws nothing while the training runs on this machine. */
export function ColabDatasetCard({
  onLocalFolder,
}: {
  /** A Drive folder was picked: it is also a folder of this computer, so the
   *  form can read its splits and classes as it does for a local dataset. */
  onLocalFolder?: (path: string) => void;
} = {}) {
  const target = useContext(ColabTargetContext);
  const t = useT();
  if (target === null) return null;
  const c = t.colab;
  const { setup, onChange, formBaseDir, problem } = target;
  const typedBadLink =
    setup.kind === "link" && setup.link.trim() !== "" && parseDatasetSource(setup.link) === null;

  return (
    <div style={CARD} data-colab-dataset={setup.kind}>
      <div style={KICKER}>{c.target.dataKicker}</div>
      <p style={DIM}>{c.target.dataIntro}</p>
      <Segmented
        label={c.target.kind}
        value={setup.kind}
        onChange={(kind) => onChange({ kind: kind === "link" ? "link" : "drive" })}
        options={[
          { value: "drive", label: c.target.kindDrive },
          { value: "link", label: c.target.kindLink },
        ]}
      />
      {setup.kind === "drive" ? (
        <>
          <DrivePicker
            driveRoot={setup.driveRoot}
            onDriveRoot={(driveRoot) => {
              saveMyDriveRoot(driveRoot);
              onChange({ driveRoot });
            }}
            folder={setup.drivePath}
            remote={setup.driveRemote}
            onPick={(drivePath, driveRemote) => {
              onChange({ drivePath, driveRemote });
              if (drivePath.trim() !== "") onLocalFolder?.(drivePath);
            }}
          />
          {setup.drivePath.trim() === "" && formBaseDir.trim() !== "" && (
            <p style={DIM}>{c.target.formFolder(formBaseDir)}</p>
          )}
          {problem === "no_folder" && (
            <p style={ERROR} data-colab-dataset-problem>
              {c.target.noFolder}
            </p>
          )}
          <p style={DIM}>{c.target.backupNote}</p>
        </>
      ) : (
        <>
          <TextField
            label={c.target.linkLabel}
            value={setup.link}
            onChange={(link) => onChange({ link })}
            placeholder={c.sourcePlaceholder}
            error={typedBadLink ? c.badSource : undefined}
            mono
          />
          <p style={DIM}>{c.target.linkClasses}</p>
        </>
      )}
      <label style={CHECK_ROW}>
        <input
          type="checkbox"
          checked={setup.driveExport}
          onChange={(e) => onChange({ driveExport: e.target.checked })}
          style={switchStyle(setup.driveExport)}
        />
        {c.driveExport}
      </label>
    </div>
  );
}
