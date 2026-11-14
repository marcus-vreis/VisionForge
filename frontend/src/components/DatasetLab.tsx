import { useEffect, useRef, useState, type KeyboardEvent as ReactKeyboardEvent, type ReactNode } from "react";
import { createPortal } from "react-dom";
import {
  ArrowLeft,
  ArrowLeftRight,
  ChartNoAxesColumn,
  ImagePlus,
  Merge,
  Shuffle,
} from "lucide-react";
import type { ExplorerForm, ExplorerSummary, ExplorerTask } from "../lib/dataset-explorer";
import {
  cocoModes,
  datasetName,
  initialSection,
  labSections,
  middleTruncate,
  stepSection,
  totalBoxes,
  trapTarget,
  type CocoMode,
  type LabSection,
} from "../lib/dataset-lab";
import { useT } from "../i18n/useT";
import { Segmented } from "./controls";
import { CocoExportPanel } from "./CocoExportPanel";
import { CocoImportPanel } from "./CocoImportPanel";
import { DatasetOverview } from "./DatasetOverview";
import { IncorporatePanel } from "./IncorporatePanel";
import { MergePanel } from "./MergePanel";
import { ResplitPanel } from "./ResplitPanel";
import { text } from "./datasetExplorerStyles";

const SECTION_ICON: Record<LabSection, ReactNode> = {
  overview: <ChartNoAxesColumn />,
  resplit: <Shuffle />,
  incorporate: <ImagePlus />,
  merge: <Merge />,
  coco: <ArrowLeftRight />,
};

const FOCUSABLE =
  'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), summary, [tabindex]:not([tabindex="-1"])';

/** Elements Tab can reach inside `root`, in order; hidden sections are skipped. */
function focusables(root: HTMLElement): HTMLElement[] {
  return Array.from(root.querySelectorAll<HTMLElement>(FOCUSABLE)).filter(
    (el) => el.getClientRects().length > 0,
  );
}

/** The COCO section: export (detection only) and import, one at a time. */
function CocoSection({
  form,
  task,
  onOpenDetectionResult,
}: {
  form: ExplorerForm | null;
  task: ExplorerTask | null;
  onOpenDetectionResult: (path: string) => void;
}) {
  const t = useT().datasetLab;
  const coco = useT().datasetCoco;
  const modes = cocoModes(task);
  const [mode, setMode] = useState<CocoMode>(modes[0]);
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
      {modes.length > 1 ? (
        <div style={{ maxWidth: 440 }}>
          <Segmented
            label={t.cocoMode}
            value={mode}
            onChange={(v) => setMode(v === "export" ? "export" : "import")}
            options={[
              { value: "export", label: coco.exportOpen },
              { value: "import", label: coco.importOpen },
            ]}
          />
        </div>
      ) : task === "classification" ? (
        <div style={text}>{t.cocoExportOnlyDetection}</div>
      ) : null}
      {form && modes.includes("export") && (
        <div hidden={mode !== "export"}>
          <CocoExportPanel form={form} />
        </div>
      )}
      <div hidden={mode !== "import"}>
        <CocoImportPanel onOpenResult={onOpenDetectionResult} />
      </div>
    </div>
  );
}

export interface DatasetLabProps {
  /** The opened dataset; null when the lab opens only to convert a COCO folder. */
  form: ExplorerForm | null;
  summary: ExplorerSummary | null;
  /** The section to show first (falls back to the first one offered). */
  initial?: LabSection;
  onClose: () => void;
  /** A tool wrote a new dataset of the same type: open it in the lab. */
  onOpenResult: (path: string) => void;
  /** COCO import wrote a YOLO detection dataset: open it in the lab. */
  onOpenDetectionResult: (path: string) => void;
}

/** The dataset lab: a full-screen dialog over the app, opened from the datasets
 *  overlay once a dataset is chosen. A rail of sections on the left (tabs on top
 *  on a narrow screen) and one tool on screen at a time. Escape and "Voltar"
 *  close it; the focus stays inside while it is open and goes back to the
 *  button that opened it. */
export function DatasetLab({
  form,
  summary,
  initial,
  onClose,
  onOpenResult,
  onOpenDetectionResult,
}: DatasetLabProps) {
  const t = useT().datasetLab;
  const task: ExplorerTask | null = form && summary ? form.task : null;
  const sections = labSections(task);
  const [section, setSection] = useState<LabSection>(() => initialSection(task, initial));
  const [visited, setVisited] = useState<LabSection[]>(() => [initialSection(task, initial)]);
  const rootRef = useRef<HTMLDivElement>(null);
  const backRef = useRef<HTMLButtonElement>(null);
  const tabRefs = useRef<Partial<Record<LabSection, HTMLButtonElement | null>>>({});
  const onCloseRef = useRef(onClose);

  useEffect(() => {
    onCloseRef.current = onClose;
  }, [onClose]);

  // Focus moves in on open and back to the opener on close; the page behind
  // does not scroll while the lab covers it.
  useEffect(() => {
    const opener = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    backRef.current?.focus();
    const overflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = overflow;
      if (opener && opener.isConnected) opener.focus();
    };
  }, []);

  // Escape closes (a field that handles its own Escape, like an open select,
  // stops it first); Tab wraps inside the dialog.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        if (e.defaultPrevented) return;
        e.preventDefault();
        onCloseRef.current();
        return;
      }
      if (e.key !== "Tab") return;
      const root = rootRef.current;
      if (!root) return;
      const items = focusables(root);
      const active = document.activeElement;
      const index = active instanceof HTMLElement ? items.indexOf(active) : -1;
      const target = trapTarget(index, items.length, e.shiftKey);
      if (target !== null) {
        e.preventDefault();
        items[target].focus();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, []);

  const go = (next: LabSection, focus = false) => {
    setSection(next);
    setVisited((v) => (v.includes(next) ? v : [...v, next]));
    if (focus) tabRefs.current[next]?.focus();
  };

  const onRailKey = (e: ReactKeyboardEvent<HTMLDivElement>) => {
    const next = stepSection(sections, section, e.key);
    if (next === null) return;
    e.preventDefault();
    go(next, true);
  };

  const name = form && summary ? datasetName(form.base_dir) : t.noDataset;
  const detection = task === "detection";

  const panels: Record<LabSection, () => ReactNode> = {
    overview: () => (form && summary ? <DatasetOverview form={form} summary={summary} /> : null),
    resplit: () => (form ? <ResplitPanel form={form} onOpenResult={onOpenResult} /> : null),
    incorporate: () => (form ? <IncorporatePanel form={form} onOpenResult={onOpenResult} /> : null),
    merge: () => (form ? <MergePanel form={form} onOpenResult={onOpenResult} /> : null),
    coco: () => (
      <CocoSection form={form} task={task} onOpenDetectionResult={onOpenDetectionResult} />
    ),
  };

  const dialog = (
    <div
      ref={rootRef}
      className="vf-lab"
      role="dialog"
      aria-modal="true"
      aria-label={t.dialogLabel(name)}
      // The overlay behind closes on a click that reaches it.
      onClick={(e) => e.stopPropagation()}
    >
      <header className="vf-lab__header">
        {/* ActionButton markup, with a ref: the focus lands here on open. */}
        <button
          ref={backRef}
          type="button"
          className="vf-action-btn vf-lab__back"
          title={t.backTitle}
          onClick={onClose}
        >
          <span className="vf-action-btn__icon" aria-hidden="true">
            <ArrowLeft />
          </span>
          <span>{t.back}</span>
        </button>
        <div className="vf-lab__id">
          <div className="vf-lab__kicker">{t.kicker}</div>
          <div className="vf-lab__name-row">
            <h1 className="vf-lab__name">{name}</h1>
            {task && (
              <span className="vf-lab__chip">
                {detection ? t.typeDetection : t.typeClassification}
              </span>
            )}
          </div>
          {form && summary ? (
            <div className="vf-lab__meta">
              <span className="vf-lab__path" title={form.base_dir}>
                {middleTruncate(form.base_dir, 72)}
              </span>
              <span className="vf-lab__totals">
                {t.images(summary.total_images)} · {t.classes(summary.class_names.length)}
                {detection ? ` · ${t.boxes(totalBoxes(summary))}` : ""}
              </span>
            </div>
          ) : (
            <div className="vf-lab__meta">
              <span style={text}>{t.noDatasetHint}</span>
            </div>
          )}
        </div>
      </header>
      <div className="vf-lab__body">
        <nav className="vf-lab__rail" aria-label={t.sectionsLabel}>
          <div role="tablist" aria-label={t.sectionsLabel} aria-orientation="vertical" onKeyDown={onRailKey}>
            {sections.map((s) => (
              <button
                key={s}
                ref={(el) => {
                  tabRefs.current[s] = el;
                }}
                type="button"
                role="tab"
                id={`vf-lab-tab-${s}`}
                aria-selected={section === s}
                aria-controls={`vf-lab-panel-${s}`}
                tabIndex={section === s ? 0 : -1}
                className="vf-lab__tab"
                onClick={() => go(s)}
              >
                <span className="vf-lab__tab-icon" aria-hidden="true">
                  {SECTION_ICON[s]}
                </span>
                <span>{t.sections[s]}</span>
              </button>
            ))}
          </div>
        </nav>
        <div className="vf-lab__content">
          {sections.map((s) =>
            visited.includes(s) ? (
              <div
                key={s}
                role="tabpanel"
                id={`vf-lab-panel-${s}`}
                aria-labelledby={`vf-lab-tab-${s}`}
                hidden={section !== s}
                className="vf-lab__panel"
              >
                {panels[s]()}
              </div>
            ) : null,
          )}
        </div>
      </div>
    </div>
  );

  // A portal: the overlay's sheet is transformed, which would pin a fixed child
  // to the sheet instead of the viewport. Into the .stage, which carries the
  // task accent tokens. Server rendering (tests) has no document.
  if (typeof document === "undefined") return dialog;
  return createPortal(dialog, document.querySelector<HTMLElement>(".stage") ?? document.body);
}
