import { useState } from "react";
import { ArrowLeftRight, FlaskConical } from "lucide-react";
import { explorerSummary, pickDatasetFolder } from "../api/client";
import {
  SPLITS,
  makeDefaultExplorerForm,
  type ExplorerForm,
  type ExplorerSummary,
  type ExplorerTask,
} from "../lib/dataset-explorer";
import { openResultForm } from "../lib/dataset-incorporate";
import { canOpenLab, datasetName, middleTruncate, summaryNotes, type SummaryNote } from "../lib/dataset-lab";
import { runPicker } from "../lib/picker-feedback";
import { useT } from "../i18n/useT";
import { ROUTE, useRouteOpen } from "../lib/session";
import { ServerFoldersNote } from "./ServerFoldersNote";
import { DatasetLab } from "./DatasetLab";
import { DatasetPicker } from "./DatasetPicker";
import { Segmented, TextField } from "./controls";
import { ActionButton } from "./ui/ActionButton";
import { card, danger, ghostButton, mono, sectionLabel, text, warn } from "./datasetExplorerStyles";

function NoteLine({ note }: { note: SummaryNote }) {
  const t = useT().datasetLab;
  const ex = useT().datasetExplorer;
  if (note.kind === "empty") return <div style={warn}>{ex.empty}</div>;
  if (note.kind === "missing_split") return <div style={warn}>{t.noteMissing(ex.splitNames[note.split])}</div>;
  return <div style={text}>{t.noteUnlabeled(note.count)}</div>;
}

/** The chosen dataset in a few lines, and the button that opens the lab. */
export function DatasetSummaryCard({
  form,
  summary,
  onOpenLab,
}: {
  form: ExplorerForm;
  summary: ExplorerSummary;
  onOpenLab: () => void;
}) {
  const t = useT().datasetLab;
  const ex = useT().datasetExplorer;
  const notes = summaryNotes(summary);
  return (
    <div style={card} data-testid="dataset-summary">
      <div style={sectionLabel}>{t.summaryLabel}</div>
      <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
        <div style={{ fontSize: 18, fontWeight: 600, color: "var(--vf-text)", wordBreak: "break-word" }}>
          {datasetName(form.base_dir)}
        </div>
        <div style={{ ...mono, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={form.base_dir}>
          {middleTruncate(form.base_dir, 64)}
        </div>
      </div>
      <div style={{ display: "flex", gap: 16, flexWrap: "wrap", ...mono, color: "var(--vf-text)" }}>
        {SPLITS.map((s) => (
          <span key={s}>
            {ex.splitNames[s]}: {summary.splits[s].present ? summary.splits[s].images : ex.splitMissing}
          </span>
        ))}
        <span>{t.classes(summary.class_names.length)}</span>
      </div>
      {notes.map((n) => (
        <NoteLine key={n.kind === "missing_split" ? `${n.kind}:${n.split}` : n.kind} note={n} />
      ))}
      {canOpenLab(summary) && (
        <div>
          <ActionButton
            icon={<FlaskConical />}
            className="vf-action-btn--primary"
            title={t.openLabHint}
            onClick={onOpenLab}
          >
            {t.openLab}
          </ActionButton>
        </div>
      )}
    </div>
  );
}

type LabState = { cocoOnly: boolean } | null;

/** The "Explorar" tab of the datasets overlay: choose the dataset (type and
 *  folder), see it in a few lines and open the dataset lab, full screen, where
 *  the overview and the tools live (ADR-165..169). */
export function DatasetExplorer() {
  const t = useT().datasetExplorer;
  const canPick = useRouteOpen(ROUTE.pickFolder);
  const lab = useT().datasetLab;
  const [form, setForm] = useState<ExplorerForm>(makeDefaultExplorerForm());
  const [opened, setOpened] = useState<ExplorerForm | null>(null);
  const [summary, setSummary] = useState<ExplorerSummary | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pickNote, setPickNote] = useState<string | null>(null);
  const [labState, setLabState] = useState<LabState>(null);

  const load = async (target: ExplorerForm): Promise<boolean> => {
    setLoading(true);
    setError(null);
    try {
      const next = await explorerSummary(target);
      setSummary(next);
      setOpened(target);
      return true;
    } catch (e) {
      setSummary(null);
      setOpened(null);
      setError(e instanceof Error ? e.message : String(e));
      return false;
    } finally {
      setLoading(false);
    }
  };

  const set = (patch: Partial<ExplorerForm>) => setForm((f) => ({ ...f, ...patch }));
  // A tool hands over the new dataset: open it, and show it in the lab.
  const openInLab = (next: ExplorerForm) => {
    setForm(next);
    void load(next).then((ok) => setLabState(ok ? { cocoOnly: false } : null));
  };
  const openResult = (path: string) => openInLab(openResultForm(opened ?? form, path));
  // An imported COCO folder becomes a YOLO detection dataset: open it as such.
  const openDetectionResult = (path: string) =>
    openInLab({ ...openResultForm(opened ?? form, path), task: "detection" });
  const canOpen = form.base_dir.trim() !== "" && !loading;
  const cocoOnly = labState?.cocoOnly ?? false;
  const showLab = labState !== null && (cocoOnly || (opened !== null && canOpenLab(summary)));

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <div style={{ maxWidth: 560 }}>
        <Segmented
          label={t.task}
          value={form.task}
          onChange={(v) => set({ task: v as ExplorerTask })}
          options={[
            { value: "classification", label: t.taskClassification },
            { value: "detection", label: t.taskDetection },
          ]}
        />
      </div>
      {form.task === "classification" ? (
        <DatasetPicker
          baseDir={form.base_dir}
          trainDir={form.train_dir}
          valDir={form.val_dir}
          testDir={form.test_dir}
          onChange={(next) => set(next)}
        />
      ) : (
        <div className="vf-lab-row">
          <div className="vf-lab-row__field">
            <TextField
              label={t.baseDir}
              value={form.base_dir}
              onChange={(v) => set({ base_dir: v })}
              placeholder={t.baseDirPlaceholder}
              mono
            />
          </div>
          <div className="vf-lab-row__actions">
            <button
              type="button"
              hidden={!canPick}
              style={ghostButton}
              title={t.browseTitle}
              onClick={() =>
                void runPicker(
                  pickDatasetFolder,
                  (path) => {
                    set({ base_dir: path });
                    setPickNote(null);
                  },
                  { canceled: t.pickCancelled, failed: t.pickFailed },
                ).then(setPickNote)
              }
            >
              {t.browse}
            </button>
          </div>
        </div>
      )}
      {/* The classification picker above lists them itself. */}
      {form.task !== "classification" && <ServerFoldersNote />}
      {pickNote && <div style={text}>{pickNote}</div>}
      <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
        <ActionButton disabled={!canOpen} onClick={() => void load(form)}>
          {summary ? t.refresh : t.open}
        </ActionButton>
        {form.task === "detection" && (
          <ActionButton
            icon={<ArrowLeftRight />}
            size="sm"
            title={lab.cocoEntryHint}
            onClick={() => setLabState({ cocoOnly: true })}
          >
            {lab.cocoEntry}
          </ActionButton>
        )}
      </div>
      {loading && <div style={text}>{t.loading}</div>}
      {error && <div style={danger}>{t.loadFailed(error)}</div>}
      {summary && opened && (
        <DatasetSummaryCard
          form={opened}
          summary={summary}
          onOpenLab={() => setLabState({ cocoOnly: false })}
        />
      )}
      {showLab && (
        <DatasetLab
          key={cocoOnly ? "coco" : `${opened?.task}:${opened?.base_dir}`}
          form={cocoOnly ? null : opened}
          summary={cocoOnly ? null : summary}
          initial={cocoOnly ? "coco" : undefined}
          onClose={() => setLabState(null)}
          onOpenResult={openResult}
          onOpenDetectionResult={openDetectionResult}
        />
      )}
    </div>
  );
}
