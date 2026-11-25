import { useRef, useState, type CSSProperties } from "react";

import {
  ApiError,
  cancelDatasetUpload,
  completeDatasetUpload,
  fetchDatasetUpload,
  sendDatasetUploadPart,
  startDatasetUpload,
} from "../api/client";
import { useI18n } from "../i18n/useT";
import {
  folderNameOf,
  percentOf,
  sendMissing,
  type DatasetUpload,
  type UploadedDataset,
} from "../lib/dataset-upload";
import { diskSize } from "../lib/session";
import { TextField } from "./controls";
import { Progress } from "./JobProgress";

/** idle: nothing on its way; sending: parts going up; finishing: the server
 *  checks and unpacks; failed: stopped, with the upload still open to resume. */
export type UploadPhase = "idle" | "sending" | "finishing" | "failed";

const BUTTON: CSSProperties = {
  padding: "7px 14px",
  borderRadius: 8,
  border: "1px solid var(--vf-panel-stroke)",
  background: "transparent",
  color: "var(--vf-text)",
  fontFamily: "var(--font-mono)",
  fontSize: 11,
  letterSpacing: "0.10em",
  textTransform: "uppercase",
  cursor: "pointer",
};

const CALLS = { sendPart: sendDatasetUploadPart, complete: completeDatasetUpload };

interface OwnDatasetUploadProps {
  /** Runs once the dataset is in the person's folder. */
  onUploaded: (dataset: UploadedDataset) => Promise<void> | void;
}

/** "Send .zip" in the person's own datasets on a shared server (ADR-246).
 *
 * The file goes up in the parts the server asks for, one after the other.
 * When a part fails the upload stays open: "resume" asks the server which
 * parts it has and sends only the rest.
 */
export function OwnDatasetUpload({ onUploaded }: OwnDatasetUploadProps) {
  const { t } = useI18n();
  const picker = useRef<HTMLInputElement>(null);
  const [name, setName] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [upload, setUpload] = useState<DatasetUpload | null>(null);
  const [sent, setSent] = useState(0);
  const [phase, setPhase] = useState<UploadPhase>("idle");
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const reset = () => {
    setUpload(null);
    setFile(null);
    setSent(0);
    setPhase("idle");
  };

  const failed = (e: unknown, keepOpen: boolean) => {
    setError(e instanceof ApiError ? e.message : t.ownDatasets.uploadFailed);
    if (keepOpen) setPhase("failed");
    else reset();
  };

  const send = async (chosen: File, open: DatasetUpload) => {
    setUpload(open);
    setPhase("sending");
    try {
      const dataset = await sendMissing(CALLS, chosen, open, setSent, () =>
        setPhase("finishing"),
      );
      reset();
      setName("");
      setNotice(t.ownDatasets.uploaded(dataset.name));
      await onUploaded(dataset);
    } catch (e) {
      failed(e, true);
    }
  };

  const begin = async (chosen: File) => {
    setError(null);
    setNotice(null);
    setFile(chosen);
    setSent(0);
    setPhase("sending");
    let open: DatasetUpload;
    try {
      open = await startDatasetUpload({
        name: name.trim() || folderNameOf(chosen.name),
        filename: chosen.name,
        size: chosen.size,
      });
    } catch (e) {
      failed(e, false);
      return;
    }
    await send(chosen, open);
  };

  const resume = async () => {
    if (file === null || upload === null) return;
    setError(null);
    setPhase("sending");
    let fresh: DatasetUpload;
    try {
      fresh = await fetchDatasetUpload(upload.id);
    } catch (e) {
      // Gone (refused when unpacked, cancelled, expired): start over.
      failed(e, !(e instanceof ApiError && e.status === 404));
      return;
    }
    await send(file, fresh);
  };

  const cancel = async () => {
    if (upload !== null) {
      try {
        await cancelDatasetUpload(upload.id);
      } catch {
        // Already gone on the server; nothing to drop.
      }
    }
    setError(null);
    reset();
  };

  return (
    <>
      <input
        ref={picker}
        type="file"
        accept=".zip,application/zip"
        style={{ display: "none" }}
        onChange={(e) => {
          const chosen = e.target.files?.[0];
          e.target.value = "";
          if (chosen) void begin(chosen);
        }}
      />
      <OwnDatasetUploadView
        phase={phase}
        name={name}
        sent={sent}
        total={file?.size ?? 0}
        error={error}
        notice={notice}
        onName={setName}
        onPick={() => picker.current?.click()}
        onResume={() => void resume()}
        onCancel={() => void cancel()}
      />
    </>
  );
}

interface OwnDatasetUploadViewProps {
  phase: UploadPhase;
  name: string;
  /** Bytes on the server, and the file's size. */
  sent: number;
  total: number;
  error: string | null;
  notice: string | null;
  onName: (name: string) => void;
  onPick: () => void;
  onResume: () => void;
  onCancel: () => void;
}

/** What the upload shows in each phase; the tests render it without a server. */
export function OwnDatasetUploadView({
  phase,
  name,
  sent,
  total,
  error,
  notice,
  onName,
  onPick,
  onResume,
  onCancel,
}: OwnDatasetUploadViewProps) {
  const { t, locale } = useI18n();
  const busy = phase === "sending" || phase === "finishing";
  const percent = percentOf(sent, total);
  return (
    <div data-own-upload={phase} style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      <p style={{ margin: 0, lineHeight: 1.55, color: "var(--vf-text-dim)" }}>
        {t.ownDatasets.uploadIntro}
      </p>
      <div style={{ display: "flex", alignItems: "flex-end", flexWrap: "wrap", gap: 10 }}>
        <div style={{ flex: "1 1 220px", minWidth: 0 }}>
          <TextField
            label={t.ownDatasets.uploadName}
            value={name}
            onChange={onName}
            hint={t.ownDatasets.uploadNameHint}
            mono
          />
        </div>
        <button
          type="button"
          style={BUTTON}
          disabled={phase !== "idle"}
          onClick={onPick}
        >
          {t.ownDatasets.upload}
        </button>
      </div>
      {busy && (
        <div role="status" style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          <Progress pct={phase === "finishing" ? 100 : percent} />
          <span style={{ fontFamily: "var(--font-mono)", fontSize: 12 }}>
            {phase === "finishing"
              ? t.ownDatasets.uploadFinishing
              : t.ownDatasets.uploadProgress(
                  percent,
                  diskSize(sent, locale),
                  diskSize(total, locale),
                )}
          </span>
        </div>
      )}
      {notice && <div role="status">{notice}</div>}
      {error && <div role="alert">{error}</div>}
      {phase === "failed" && (
        <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
          <button type="button" style={BUTTON} onClick={onResume}>
            {t.ownDatasets.uploadResume}
          </button>
          <button type="button" style={BUTTON} onClick={onCancel}>
            {t.ownDatasets.uploadCancel}
          </button>
        </div>
      )}
    </div>
  );
}
