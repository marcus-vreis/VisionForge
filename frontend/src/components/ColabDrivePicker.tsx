/** Choosing the Colab dataset folder inside this computer's "My Drive" (ADR-164).
 *
 * The dataset lives in the local Google Drive for desktop folder; the server
 * lists its sub-folders one level at a time and says where the chosen one is on
 * the VM. Without Drive for desktop, the picker explains how to get it instead.
 * Nothing here talks to Google: browsing Drive on the web is out of scope.
 */

import { ChevronRight, Folder, FolderOpen } from "lucide-react";
import { useCallback, useEffect, useRef, useState, type CSSProperties, type ReactNode, type RefObject } from "react";
import { fetchDriveFolders, openDriveFolder, type ColabDriveFolders } from "../api/client";
import { useT } from "../i18n/useT";
import { childRel, driveCrumbs } from "../lib/drive-folders";
import { ACTION, BODY, DIM, ERROR, FIELD, LABEL, LINK, STACK } from "./colabStyles";

const SECONDARY: CSSProperties = {
  ...ACTION,
  display: "inline-flex",
  alignItems: "center",
  gap: 8,
  border: "1px solid var(--vf-panel-stroke)",
};

const BOX: CSSProperties = {
  ...STACK,
  gap: 10,
  padding: 12,
  borderRadius: 10,
  border: "1px solid var(--vf-panel-stroke)",
  background: "rgb(var(--lt-shade, 0 0 0) / 0.2)",
};

const ROW: CSSProperties = {
  ...BODY,
  display: "flex",
  alignItems: "center",
  gap: 8,
  width: "100%",
  padding: "7px 9px",
  border: "none",
  borderRadius: 7,
  background: "transparent",
  textAlign: "left",
  cursor: "pointer",
};

const CRUMB: CSSProperties = {
  ...LINK,
  textDecoration: "none",
  maxWidth: 160,
  overflow: "hidden",
  textOverflow: "ellipsis",
  whiteSpace: "nowrap",
};

/** What the server last said about "My Drive", and whether a request is out. */
export interface DriveView {
  listing: ColabDriveFolders | null;
  loading: boolean;
  error: string | null;
}

export interface DrivePickerViewProps {
  view: DriveView;
  browsing: boolean;
  /** The dataset folder on this machine (the field's text). */
  folder: string;
  /** Where `folder` is on the VM; known only for a folder picked in the browser. */
  remote: string | null;
  driveRoot: string;
  onFolder: (path: string) => void;
  onDriveRoot: (root: string) => void;
  onRetry: () => void;
  onBrowse: () => void;
  onEnter: (rel: string) => void;
  onUse: () => void;
  onCancelBrowse: () => void;
  onOpen: () => void;
  boxRef?: RefObject<HTMLDivElement | null>;
}

function DriveRootField({
  value,
  busy,
  onChange,
  onRetry,
}: {
  value: string;
  busy: boolean;
  onChange: (root: string) => void;
  onRetry: () => void;
}) {
  const c = useT().colab;
  return (
    <div style={{ ...STACK, gap: 6 }}>
      <label style={LABEL} htmlFor="colab-drive-root">
        {c.driveRoot}
      </label>
      <div style={{ display: "flex", gap: 8 }}>
        <input
          id="colab-drive-root"
          type="text"
          autoComplete="off"
          spellCheck={false}
          value={value}
          placeholder={c.driveRootPlaceholder}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && onRetry()}
          style={{ ...FIELD, flex: 1 }}
        />
        <button type="button" style={SECONDARY} disabled={busy} onClick={onRetry}>
          {c.driveRetry}
        </button>
      </div>
    </div>
  );
}

function Problem({ text }: { text: string | null }) {
  return text ? (
    <p role="alert" style={ERROR}>
      {text}
    </p>
  ) : null;
}

function DriveMissing({ error, rootField }: { error: string | null; rootField: ReactNode }) {
  const c = useT().colab;
  return (
    <div style={BOX} data-colab-drive="missing">
      <p style={{ ...BODY, fontWeight: 600 }}>{c.driveMissingTitle}</p>
      <ol style={{ ...BODY, margin: 0, paddingLeft: 22, lineHeight: 1.7 }}>
        <li>{c.driveStep1}</li>
        <li>{c.driveStep2}</li>
        <li>{c.driveStep3}</li>
      </ol>
      <Problem text={error} />
      {rootField}
    </div>
  );
}

function FolderRow({ name, disabled, onClick }: { name: string; disabled: boolean; onClick: () => void }) {
  return (
    <li>
      <button
        type="button"
        data-drive-folder
        disabled={disabled}
        onClick={onClick}
        style={ROW}
        onMouseEnter={(e) => (e.currentTarget.style.background = "rgb(var(--lt-ink, 255 255 255) / 0.05)")}
        onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
      >
        <Folder size={15} aria-hidden style={{ flexShrink: 0, color: "var(--vf-text-dim)" }} />
        <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{name}</span>
      </button>
    </li>
  );
}

interface DriveBrowserProps {
  listing: ColabDriveFolders;
  view: DriveView;
  boxRef?: RefObject<HTMLDivElement | null>;
  onEnter: (rel: string) => void;
  onUse: () => void;
  onCancelBrowse: () => void;
  onOpen: () => void;
}

function DriveBrowser({ listing, view, boxRef, onEnter, onUse, onCancelBrowse, onOpen }: DriveBrowserProps) {
  const c = useT().colab;
  const crumbs = driveCrumbs(listing.rel);
  return (
    <div ref={boxRef} tabIndex={-1} role="group" aria-label={c.driveBrowserLabel} style={BOX} data-colab-drive-browser>
      <nav aria-label={c.driveBrowserLabel}>
        <ol style={{ listStyle: "none", display: "flex", flexWrap: "wrap", alignItems: "center", gap: 4, margin: 0, padding: 0 }}>
          <li>
            <button
              type="button"
              style={{ ...CRUMB, color: crumbs.length ? "var(--vf-text-dim)" : "var(--vf-text)" }}
              aria-current={crumbs.length ? undefined : "location"}
              onClick={() => onEnter("")}
            >
              {c.driveHome}
            </button>
          </li>
          {crumbs.map((crumb, i) => {
            const last = i === crumbs.length - 1;
            return (
              <li key={crumb.rel} style={{ display: "flex", alignItems: "center", gap: 4 }}>
                <ChevronRight size={12} aria-hidden style={{ color: "var(--vf-text-muted)" }} />
                <button
                  type="button"
                  style={{ ...CRUMB, color: last ? "var(--vf-text)" : "var(--vf-text-dim)" }}
                  aria-current={last ? "location" : undefined}
                  onClick={() => onEnter(crumb.rel)}
                >
                  {crumb.name}
                </button>
              </li>
            );
          })}
        </ol>
      </nav>
      <div aria-live="polite">
        {view.loading && <p style={DIM}>{c.driveLoading}</p>}
        <Problem text={view.error} />
      </div>
      {listing.folders.length === 0 ? (
        <p style={DIM}>{c.driveEmpty}</p>
      ) : (
        <ul
          aria-busy={view.loading}
          style={{ listStyle: "none", margin: 0, padding: 0, maxHeight: 220, overflowY: "auto", display: "flex", flexDirection: "column", gap: 2 }}
        >
          {listing.folders.map((name) => (
            <FolderRow
              key={name}
              name={name}
              disabled={view.loading}
              onClick={() => onEnter(childRel(listing.rel, name))}
            />
          ))}
        </ul>
      )}
      {listing.truncated && <p style={DIM}>{c.driveTruncated(listing.folders.length)}</p>}
      {listing.remote && <p style={{ ...DIM, wordBreak: "break-all" }}>{c.driveRemote(listing.remote)}</p>}
      <div style={{ display: "flex", gap: 14, alignItems: "center", flexWrap: "wrap" }}>
        <button type="button" data-drive-use style={ACTION} disabled={view.loading} onClick={onUse}>
          {c.driveUse}
        </button>
        <button type="button" style={LINK} onClick={onCancelBrowse}>
          {c.cancel}
        </button>
        {listing.can_open && (
          <button type="button" style={{ ...LINK, marginLeft: "auto" }} onClick={onOpen}>
            {c.driveOpen}
          </button>
        )}
      </div>
    </div>
  );
}

/** The Drive part of the train form, for each state the server reports. */
export function DrivePickerView(props: DrivePickerViewProps) {
  const c = useT().colab;
  const { listing, loading, error } = props.view;
  const rootField = (
    <DriveRootField value={props.driveRoot} busy={loading} onChange={props.onDriveRoot} onRetry={props.onRetry} />
  );

  if (listing === null) {
    return (
      <div style={STACK} data-colab-drive={loading ? "checking" : "failed"}>
        {loading ? (
          <p role="status" style={DIM}>
            {c.driveChecking}
          </p>
        ) : (
          <Problem text={error} />
        )}
        {!loading && rootField}
      </div>
    );
  }
  if (listing.root === null) return <DriveMissing error={error} rootField={rootField} />;

  return (
    <div style={STACK} data-colab-drive="found">
      <label style={LABEL} htmlFor="colab-drive-folder">
        {c.driveFolder}
      </label>
      <input
        id="colab-drive-folder"
        type="text"
        autoComplete="off"
        spellCheck={false}
        value={props.folder}
        placeholder={c.driveFolderPlaceholder}
        onChange={(e) => props.onFolder(e.target.value)}
        style={FIELD}
      />
      {props.remote && (
        <p style={{ ...DIM, wordBreak: "break-all" }} data-colab-drive-remote>
          {c.driveRemote(props.remote)}
        </p>
      )}
      {props.browsing ? (
        <DriveBrowser
          listing={listing}
          view={props.view}
          boxRef={props.boxRef}
          onEnter={props.onEnter}
          onUse={props.onUse}
          onCancelBrowse={props.onCancelBrowse}
          onOpen={props.onOpen}
        />
      ) : (
        <>
          <div style={{ display: "flex", gap: 14, alignItems: "center", flexWrap: "wrap" }}>
            <button type="button" style={SECONDARY} disabled={loading} onClick={props.onBrowse}>
              <FolderOpen size={14} aria-hidden />
              {c.drivePick}
            </button>
            {listing.can_open && (
              <button type="button" style={LINK} onClick={props.onOpen}>
                {c.driveOpen}
              </button>
            )}
          </div>
          <Problem text={error} />
        </>
      )}
      <details>
        <summary style={{ ...LINK, display: "list-item", textDecoration: "none" }}>{c.driveRootOther}</summary>
        <div style={{ paddingTop: 8 }}>{rootField}</div>
      </details>
    </div>
  );
}

const message = (e: unknown) => (e instanceof Error ? e.message : String(e));

export interface DrivePickerProps {
  driveRoot: string;
  onDriveRoot: (root: string) => void;
  folder: string;
  remote: string | null;
  /** A folder and, when it came from the browser, its path on the VM. */
  onPick: (folder: string, remote: string | null) => void;
}

/** The picker with its requests: asks the server for "My Drive" once shown. */
export function DrivePicker({ driveRoot, onDriveRoot, folder, remote, onPick }: DrivePickerProps) {
  const [view, setView] = useState<DriveView>({ listing: null, loading: true, error: null });
  const [browsing, setBrowsing] = useState(false);
  const [firstRoot] = useState(driveRoot);
  const asked = useRef(0);
  const focusAfter = useRef(false);
  const boxRef = useRef<HTMLDivElement>(null);

  // Only the latest request lands: a slow answer for a folder left behind
  // must not replace the one the person is looking at.
  const request = useCallback((rel: string, root: string) => {
    const id = ++asked.current;
    fetchDriveFolders(rel, root).then(
      (listing) => {
        if (id === asked.current) setView({ listing, loading: false, error: null });
      },
      (e: unknown) => {
        if (id === asked.current) setView((v) => ({ ...v, loading: false, error: message(e) }));
      },
    );
  }, []);

  useEffect(() => {
    request("", firstRoot);
    return () => {
      asked.current += 1;
    };
  }, [request, firstRoot]);

  // After entering a folder, keyboard focus goes to its first sub-folder.
  useEffect(() => {
    if (!focusAfter.current) return;
    focusAfter.current = false;
    const box = boxRef.current;
    const target = box?.querySelector<HTMLButtonElement>("[data-drive-folder]") ?? box;
    target?.focus();
  }, [view.listing]);

  const load = (rel: string) => {
    setView((v) => ({ ...v, loading: true, error: null }));
    request(rel, driveRoot);
  };

  return (
    <DrivePickerView
      view={view}
      browsing={browsing}
      folder={folder}
      remote={remote}
      driveRoot={driveRoot}
      boxRef={boxRef}
      onFolder={(path) => onPick(path, null)}
      onDriveRoot={onDriveRoot}
      onRetry={() => {
        setBrowsing(false);
        load("");
      }}
      onBrowse={() => {
        setBrowsing(true);
        focusAfter.current = true;
        load(view.listing?.rel ?? "");
      }}
      onEnter={(rel) => {
        focusAfter.current = true;
        load(rel);
      }}
      onUse={() => {
        const listing = view.listing;
        if (listing?.path) onPick(listing.path, listing.remote);
        setBrowsing(false);
      }}
      onCancelBrowse={() => setBrowsing(false)}
      onOpen={() => {
        const rel = browsing ? (view.listing?.rel ?? "") : "";
        openDriveFolder(rel, driveRoot).catch((e: unknown) =>
          setView((v) => ({ ...v, error: message(e) })),
        );
      }}
    />
  );
}
