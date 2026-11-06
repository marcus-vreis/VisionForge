import { useCallback, useEffect, useState } from "react";
import {
  explorerImages,
  explorerScanCancel,
  explorerScanStart,
  explorerScanStatus,
  explorerSummary,
  pickDatasetFolder,
  type ExplorerImagesPage,
} from "../api/client";
import {
  GALLERY_PAGE_SIZE,
  SPLITS,
  barsOf,
  binBars,
  classSplitRows,
  hasProblems,
  isScanFinished,
  makeDefaultExplorerForm,
  pagingOf,
  scanPercent,
  stepOffset,
  thumbUrl,
  type Bar,
  type DuplicateGroup,
  type ExplorerForm,
  type ExplorerSummary,
  type ExplorerTask,
  type ScanJob,
  type ScanReport,
} from "../lib/dataset-explorer";
import { openResultForm } from "../lib/dataset-incorporate";
import { runPicker } from "../lib/picker-feedback";
import { useT } from "../i18n/useT";
import { DatasetPicker } from "./DatasetPicker";
import { Segmented, SelectField, TextField } from "./controls";
import { button, card, ghostButton, mono, sectionLabel } from "./datasetExplorerStyles";
import { IncorporatePanel } from "./IncorporatePanel";

function BarList({ bars }: { bars: Bar[] }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      {bars.map((b) => (
        <div
          key={b.label}
          style={{
            display: "grid",
            gridTemplateColumns: "minmax(70px, 160px) 1fr 56px",
            alignItems: "center",
            gap: 8,
            ...mono,
          }}
        >
          <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
            {b.label}
          </span>
          <span
            style={{
              height: 8,
              borderRadius: 4,
              background: "rgb(var(--lt-ink, 255 255 255) / 0.06)",
              overflow: "hidden",
            }}
          >
            <span
              style={{
                display: "block",
                height: "100%",
                width: `${b.pct}%`,
                background: "var(--accent-vf)",
              }}
            />
          </span>
          <span style={{ textAlign: "right" }}>{b.value}</span>
        </div>
      ))}
    </div>
  );
}

function Distribution({ summary }: { summary: ExplorerSummary }) {
  const t = useT().datasetExplorer;
  const detection = summary.task === "detection";
  const rows = classSplitRows(summary, "per_class");
  const boxRows = detection ? classSplitRows(summary, "boxes") : [];
  return (
    <div style={card}>
      <div style={sectionLabel}>{t.distributionTitle}</div>
      <div style={{ display: "flex", gap: 18, flexWrap: "wrap", ...mono }}>
        <span>{t.totalImages(summary.total_images)}</span>
        {SPLITS.map((s) => {
          const sp = summary.splits[s];
          return (
            <span key={s}>
              {t.splitNames[s]}:{" "}
              {sp.present ? sp.images : t.splitMissing}
              {detection && sp.present && sp.unlabeled > 0
                ? ` (${t.unlabeled(sp.unlabeled)})`
                : ""}
            </span>
          );
        })}
      </div>
      <div style={sectionLabel}>{t.imagesPerClass}</div>
      <table style={{ ...mono, borderCollapse: "collapse", width: "100%" }}>
        <thead>
          <tr>
            <th style={{ textAlign: "left" }}>{t.classColumn}</th>
            {SPLITS.map((s) => (
              <th key={s} style={{ textAlign: "right" }}>
                {t.splitNames[s]}
              </th>
            ))}
            <th style={{ textAlign: "right" }}>{t.totalColumn}</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.name}>
              <td>{r.name}</td>
              {SPLITS.map((s) => (
                <td key={s} style={{ textAlign: "right" }}>
                  {r.counts[s]}
                </td>
              ))}
              <td style={{ textAlign: "right" }}>{r.total}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {detection && (
        <>
          <div style={sectionLabel}>{t.boxesPerClass}</div>
          <BarList
            bars={barsOf(Object.fromEntries(boxRows.map((r) => [r.name, r.total])))}
          />
        </>
      )}
    </div>
  );
}

function Gallery({
  form,
  summary,
}: {
  form: ExplorerForm;
  summary: ExplorerSummary;
}) {
  const t = useT().datasetExplorer;
  const [split, setSplit] = useState("");
  const [className, setClassName] = useState("");
  const [offset, setOffset] = useState(0);
  const [data, setData] = useState<ExplorerImagesPage | null>(null);
  const [error, setError] = useState<string | null>(null);

  // A new summary (another folder, or a refresh) restarts the gallery.
  useEffect(() => {
    const timer = setTimeout(() => setOffset(0), 0);
    return () => clearTimeout(timer);
  }, [summary]);

  useEffect(() => {
    let alive = true;
    explorerImages(
      form,
      { split: split || null, className: className || null },
      offset,
      GALLERY_PAGE_SIZE,
    )
      .then((page) => {
        if (!alive) return;
        setData(page);
        setError(null);
      })
      .catch((e: unknown) => {
        if (alive) setError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      alive = false;
    };
  }, [form, summary, split, className, offset]);

  const total = data?.total ?? 0;
  const paging = pagingOf(total, GALLERY_PAGE_SIZE, offset);
  const splitOptions = [
    { value: "", label: t.all },
    ...SPLITS.filter((s) => summary.splits[s].present).map((s) => ({
      value: s,
      label: t.splitNames[s],
    })),
  ];
  const classOptions = [
    { value: "", label: t.all },
    ...summary.class_names.map((c) => ({ value: c, label: c })),
  ];
  const root = summary.base_dir;

  return (
    <div style={card}>
      <div style={sectionLabel}>{t.galleryTitle}</div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
        <SelectField
          label={t.filterSplit}
          value={split}
          onChange={(v) => {
            setSplit(v);
            setOffset(0);
          }}
          options={splitOptions}
        />
        <SelectField
          label={t.filterClass}
          value={className}
          onChange={(v) => {
            setClassName(v);
            setOffset(0);
          }}
          options={classOptions}
        />
      </div>
      {error && <div style={{ ...mono, color: "var(--vf-danger, #f87171)" }}>{error}</div>}
      {data && data.items.length === 0 && <div style={mono}>{t.noMatch}</div>}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(112px, 1fr))",
          gap: 8,
        }}
      >
        {data?.items.map((it) => (
          <figure key={`${it.split}/${it.path}`} style={{ margin: 0 }}>
            <img
              src={thumbUrl(root, it.path, 160, it.label_path)}
              alt={it.path}
              loading="lazy"
              style={{
                width: "100%",
                aspectRatio: "1",
                objectFit: "cover",
                borderRadius: 8,
                background: "rgb(var(--lt-ink, 255 255 255) / 0.05)",
              }}
            />
            <figcaption style={{ ...mono, fontSize: 10, wordBreak: "break-all" }}>
              {t.splitNames[it.split as keyof typeof t.splitNames]} · {it.classes.join(", ")}
              {it.boxes > 0 ? ` · ${t.boxesOnImage(it.boxes)}` : ""}
            </figcaption>
          </figure>
        ))}
      </div>
      {total > 0 && (
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <button
            type="button"
            style={ghostButton}
            disabled={paging.page === 0}
            onClick={() => setOffset(stepOffset(total, GALLERY_PAGE_SIZE, offset, -1))}
          >
            {t.previous}
          </button>
          <span style={mono}>{t.range(paging.from, paging.to, total)}</span>
          <button
            type="button"
            style={ghostButton}
            disabled={paging.page >= paging.pages - 1}
            onClick={() => setOffset(stepOffset(total, GALLERY_PAGE_SIZE, offset, 1))}
          >
            {t.next}
          </button>
        </div>
      )}
    </div>
  );
}

function GroupList({
  title,
  help,
  groups,
  total,
}: {
  title: string;
  help?: string;
  groups: DuplicateGroup[];
  total: number;
}) {
  const t = useT().datasetExplorer;
  if (total === 0) return null;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      <div style={sectionLabel}>
        {title} ({total})
      </div>
      {help && <div style={mono}>{help}</div>}
      {groups.map((g) => (
        <div key={g.hash} style={{ ...mono, paddingLeft: 10, borderLeft: "2px solid var(--vf-panel-stroke)" }}>
          <div>
            {t.groupFiles(g.files.length)}
            {g.label_conflict ? ` · ${t.labelConflict}` : ""}
          </div>
          {g.files.map((f) => (
            <div key={`${f.split}/${f.path}`} style={{ wordBreak: "break-all" }}>
              [{t.splitNames[f.split as keyof typeof t.splitNames]}] {f.path}
            </div>
          ))}
        </div>
      ))}
      {total > groups.length && <div style={mono}>{t.moreNotShown(total - groups.length)}</div>}
    </div>
  );
}

function Report({ report }: { report: ScanReport }) {
  const t = useT().datasetExplorer;
  const { sizes } = report;
  return (
    <>
      <div style={{ ...mono, color: report.sampled || report.cancelled ? "var(--vf-warn, #fbbf24)" : undefined }}>
        {report.cancelled
          ? t.scanCancelled(report.scanned)
          : report.sampled
            ? t.scanSampled(report.scanned, report.total_images)
            : t.scanComplete(report.scanned)}
      </div>
      <div style={card}>
        <div style={sectionLabel}>{t.problemsTitle}</div>
        {!hasProblems(report) && <div style={mono}>{t.noProblems}</div>}
        {report.corrupted_count > 0 && (
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <div style={sectionLabel}>
              {t.corruptedTitle} ({report.corrupted_count})
            </div>
            {report.corrupted.map((c) => (
              <div key={`${c.split}/${c.path}`} style={{ ...mono, wordBreak: "break-all" }}>
                [{t.splitNames[c.split as keyof typeof t.splitNames]}] {c.path} · {c.reason}
              </div>
            ))}
            {report.corrupted_count > report.corrupted.length && (
              <div style={mono}>
                {t.moreNotShown(report.corrupted_count - report.corrupted.length)}
              </div>
            )}
          </div>
        )}
        <GroupList
          title={t.leakageTitle}
          help={t.leakageHelp}
          groups={report.leakage_groups}
          total={report.leakage_group_count}
        />
        <GroupList
          title={t.duplicatesTitle}
          groups={report.duplicate_groups}
          total={report.duplicate_group_count}
        />
      </div>
      {sizes.count > 0 && sizes.width && sizes.height && (
        <div style={card}>
          <div style={sectionLabel}>{t.sizesTitle}</div>
          <div style={mono}>
            <div>{t.widthStats(sizes.width.min, sizes.width.median, sizes.width.max)}</div>
            <div>{t.heightStats(sizes.height.min, sizes.height.median, sizes.height.max)}</div>
            <div>{t.distinctSizes(sizes.distinct_sizes ?? 0)}</div>
          </div>
          {sizes.top_sizes && (
            <div style={mono}>
              {t.topSizes}: {sizes.top_sizes.map((s) => t.sizeCount(s.width, s.height, s.count)).join("  ·  ")}
            </div>
          )}
          {sizes.side_bins && (
            <>
              <div style={sectionLabel}>{t.longestSide}</div>
              <BarList bars={binBars(sizes.side_bins)} />
            </>
          )}
          {sizes.aspect_bins && (
            <>
              <div style={sectionLabel}>{t.aspectRatio}</div>
              <BarList bars={binBars(sizes.aspect_bins, 2)} />
            </>
          )}
        </div>
      )}
    </>
  );
}

function ScanPanel({ form }: { form: ExplorerForm }) {
  const t = useT().datasetExplorer;
  const [limit, setLimit] = useState("20000");
  const [job, setJob] = useState<ScanJob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [cancelling, setCancelling] = useState(false);
  const jobId = job?.job_id ?? null;
  const running = job !== null && !isScanFinished(job.state);

  useEffect(() => {
    if (!jobId || !running) return;
    const timer = setInterval(() => {
      explorerScanStatus(jobId)
        .then((next) => {
          setJob(next);
          if (isScanFinished(next.state)) setCancelling(false);
        })
        .catch((e: unknown) => {
          setError(e instanceof Error ? e.message : String(e));
          setCancelling(false);
        });
    }, 600);
    return () => clearInterval(timer);
  }, [jobId, running]);

  const start = useCallback(async () => {
    setError(null);
    setJob(null);
    const n = Math.max(1, Math.floor(Number(limit)) || 20000);
    try {
      setJob(await explorerScanStart(form, n));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [form, limit]);

  const cancel = async () => {
    if (!jobId) return;
    setCancelling(true);
    try {
      await explorerScanCancel(jobId);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setCancelling(false);
    }
  };

  const pct = job ? scanPercent(job.done, job.total) : 0;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <div style={card}>
        <div style={sectionLabel}>{t.scanTitle}</div>
        <div style={mono}>{t.scanHelp}</div>
        <div style={{ display: "flex", alignItems: "flex-end", gap: 12 }}>
          <div style={{ width: 180 }}>
            <TextField
              label={t.scanLimit}
              value={limit}
              onChange={setLimit}
              mono
              hint={t.scanLimitHint}
            />
          </div>
          {running ? (
            <button type="button" style={ghostButton} onClick={() => void cancel()} disabled={cancelling}>
              {cancelling ? t.scanCancelling : t.scanCancel}
            </button>
          ) : (
            <button type="button" style={button} onClick={() => void start()}>
              {t.scanStart}
            </button>
          )}
        </div>
        {running && job && (
          <div>
            <div style={mono}>{t.scanProgress(job.done, job.total)}</div>
            <div
              role="progressbar"
              aria-valuenow={pct}
              aria-valuemin={0}
              aria-valuemax={100}
              style={{ height: 6, borderRadius: 3, background: "rgb(var(--lt-ink, 255 255 255) / 0.08)" }}
            >
              <div style={{ height: "100%", width: `${pct}%`, borderRadius: 3, background: "var(--accent-vf)" }} />
            </div>
          </div>
        )}
        {error && <div style={{ ...mono, color: "var(--vf-danger, #f87171)" }}>{t.scanFailed(error)}</div>}
        {job?.state === "failed" && job.error && (
          <div style={{ ...mono, color: "var(--vf-danger, #f87171)" }}>{t.scanFailed(job.error)}</div>
        )}
      </div>
      {job?.result && <Report report={job.result} />}
    </div>
  );
}

/** The read-only dataset explorer (ADR-165): distribution, a filterable gallery
 *  and an integrity scan for classification (ImageFolder) and detection (YOLO). */
export function DatasetExplorer() {
  const t = useT().datasetExplorer;
  const [form, setForm] = useState<ExplorerForm>(makeDefaultExplorerForm());
  const [opened, setOpened] = useState<ExplorerForm | null>(null);
  const [summary, setSummary] = useState<ExplorerSummary | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pickNote, setPickNote] = useState<string | null>(null);

  const load = async (target: ExplorerForm) => {
    setLoading(true);
    setError(null);
    try {
      const next = await explorerSummary(target);
      setSummary(next);
      setOpened(target);
    } catch (e) {
      setSummary(null);
      setOpened(null);
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  const set = (patch: Partial<ExplorerForm>) => setForm((f) => ({ ...f, ...patch }));
  // The incorporate panel hands over the new dataset: open it like any other.
  const openResult = (path: string) => {
    const next = openResultForm(opened ?? form, path);
    setForm(next);
    void load(next);
  };
  const canOpen = form.base_dir.trim() !== "" && !loading;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <div style={mono}>{t.intro}</div>
      <div style={{ maxWidth: 420 }}>
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
        <div style={{ display: "flex", alignItems: "flex-end", gap: 12 }}>
          <div style={{ flex: 1 }}>
            <TextField
              label={t.baseDir}
              value={form.base_dir}
              onChange={(v) => set({ base_dir: v })}
              placeholder={t.baseDirPlaceholder}
              mono
            />
          </div>
          <button
            type="button"
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
      )}
      {pickNote && <div style={mono}>{pickNote}</div>}
      <div style={{ display: "flex", gap: 10 }}>
        <button type="button" style={button} disabled={!canOpen} onClick={() => void load(form)}>
          {summary ? t.refresh : t.open}
        </button>
      </div>
      {loading && <div style={mono}>{t.loading}</div>}
      {error && <div style={{ ...mono, color: "var(--vf-danger, #f87171)" }}>{t.loadFailed(error)}</div>}
      {summary && opened && (
        <>
          {summary.total_images === 0 ? (
            <div style={mono}>{t.empty}</div>
          ) : (
            <>
              <Distribution summary={summary} />
              <IncorporatePanel form={opened} onOpenResult={openResult} />
              <Gallery form={opened} summary={summary} />
              <ScanPanel form={opened} />
            </>
          )}
        </>
      )}
    </div>
  );
}
