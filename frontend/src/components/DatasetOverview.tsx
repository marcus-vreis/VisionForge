import { useCallback, useEffect, useState } from "react";
import {
  explorerImages,
  explorerScanCancel,
  explorerScanStart,
  explorerScanStatus,
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
  pagingOf,
  scanPercent,
  stepOffset,
  thumbUrl,
  type Bar,
  type DuplicateGroup,
  type ExplorerForm,
  type ExplorerSummary,
  type ScanJob,
  type ScanReport,
  type SplitName,
} from "../lib/dataset-explorer";
import { classStacks, totalBoxes } from "../lib/dataset-lab";
import { useT } from "../i18n/useT";
import { SelectField, TextField } from "./controls";
import { ActionButton } from "./ui/ActionButton";
import { DataTable, LearnMore } from "./DatasetLabParts";
import {
  card,
  danger,
  ghostButton,
  mono,
  sectionLabel,
  stepTitle,
  text,
  warn,
} from "./datasetExplorerStyles";

/** Fill of each split's segment in a stacked bar: one accent, three strengths. */
const SPLIT_FILL: Record<SplitName, string> = {
  train: "var(--accent-vf)",
  val: "color-mix(in oklch, var(--accent-vf) 55%, transparent)",
  test: "color-mix(in oklch, var(--accent-vf) 28%, transparent)",
};

const track = "rgb(var(--lt-ink, 255 255 255) / 0.06)";

function CardTitle({ children }: { children: string }) {
  return <h3 style={{ ...stepTitle, margin: 0 }}>{children}</h3>;
}

/** Horizontal bars, the exact count at the end of each. */
export function BarList({ bars }: { bars: Bar[] }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      {bars.map((b) => (
        <div key={b.label} className="vf-lab-bar">
          <span className="vf-lab-bar__label" title={b.label}>
            {b.label}
          </span>
          <span className="vf-lab-bar__track" style={{ background: track }}>
            <span style={{ display: "block", height: "100%", width: `${b.pct}%`, background: "var(--accent-vf)" }} />
          </span>
          <span className="vf-lab-bar__value">{b.value}</span>
        </div>
      ))}
    </div>
  );
}

/** The dataset's totals as cards: all images, classes, then one card per split. */
function TotalCards({ summary }: { summary: ExplorerSummary }) {
  const t = useT().datasetLab;
  const ex = useT().datasetExplorer;
  const detection = summary.task === "detection";
  const cards: { key: string; label: string; value: string; note?: string; missing?: boolean }[] = [
    { key: "total", label: t.cardTotal, value: String(summary.total_images) },
    { key: "classes", label: t.cardClasses, value: String(summary.class_names.length) },
  ];
  if (detection) cards.push({ key: "boxes", label: t.cardBoxes, value: String(totalBoxes(summary)) });
  for (const s of SPLITS) {
    const sp = summary.splits[s];
    cards.push({
      key: s,
      label: ex.splitNames[s],
      value: sp.present ? String(sp.images) : "—",
      note: !sp.present
        ? t.cardMissing
        : detection && sp.unlabeled > 0
          ? t.cardUnlabeled(sp.unlabeled)
          : undefined,
      missing: !sp.present,
    });
  }
  return (
    <div className="vf-lab-cards vf-lab-cards--totals">
      {cards.map((c) => (
        <div key={c.key} className="vf-lab-stat">
          <div style={sectionLabel}>{c.label}</div>
          <div className="vf-lab-stat__value">{c.value}</div>
          {c.note && <div style={c.missing ? warn : text}>{c.note}</div>}
        </div>
      ))}
    </div>
  );
}

/** Images per class as stacked bars (one segment per split), the class total at
 *  the end, and the exact class x split table on demand. */
function Distribution({ summary }: { summary: ExplorerSummary }) {
  const t = useT().datasetLab;
  const ex = useT().datasetExplorer;
  const detection = summary.task === "detection";
  const stacks = classStacks(summary);
  const rows = classSplitRows(summary, "per_class");
  const boxRows = detection ? classSplitRows(summary, "boxes") : [];
  return (
    <div style={card}>
      <CardTitle>{t.perClassTitle}</CardTitle>
      <div style={text}>{t.perClassHelp}</div>
      <div className="vf-lab-legend" aria-hidden="true">
        {SPLITS.filter((s) => summary.splits[s].present).map((s) => (
          <span key={s} className="vf-lab-legend__item">
            <span className="vf-lab-legend__swatch" style={{ background: SPLIT_FILL[s] }} />
            {ex.splitNames[s]}
          </span>
        ))}
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        {stacks.map((r) => (
          <div key={r.name} className="vf-lab-bar">
            <span className="vf-lab-bar__label" title={r.name}>
              {r.name}
            </span>
            <span
              className="vf-lab-bar__track"
              style={{ background: track, display: "flex" }}
              role="img"
              aria-label={SPLITS.map((s) => t.segmentTitle(ex.splitNames[s], r.counts[s])).join(", ")}
            >
              {SPLITS.map((s) =>
                r.pct[s] > 0 ? (
                  <span
                    key={s}
                    title={t.segmentTitle(ex.splitNames[s], r.counts[s])}
                    style={{ display: "block", height: "100%", width: `${r.pct[s]}%`, background: SPLIT_FILL[s] }}
                  />
                ) : null,
              )}
            </span>
            <span className="vf-lab-bar__value">{r.total}</span>
          </div>
        ))}
      </div>
      <LearnMore summary={t.tableToggle}>
        <DataTable>
          <thead>
            <tr>
              <th style={{ textAlign: "left" }}>{ex.classColumn}</th>
              {SPLITS.map((s) => (
                <th key={s} style={{ textAlign: "right" }}>
                  {ex.splitNames[s]}
                </th>
              ))}
              <th style={{ textAlign: "right" }}>{ex.totalColumn}</th>
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
        </DataTable>
      </LearnMore>
      {detection && (
        <>
          <CardTitle>{t.boxesPerClassTitle}</CardTitle>
          <BarList bars={barsOf(Object.fromEntries(boxRows.map((r) => [r.name, r.total])))} />
        </>
      )}
    </div>
  );
}

function Gallery({ form, summary }: { form: ExplorerForm; summary: ExplorerSummary }) {
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
      <CardTitle>{t.galleryTitle}</CardTitle>
      <div className="vf-lab-fields">
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
      {error && <div style={danger}>{error}</div>}
      {data && data.items.length === 0 && <div style={text}>{t.noMatch}</div>}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(120px, 1fr))",
          gap: 10,
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
            <figcaption style={{ ...mono, fontSize: 11, wordBreak: "break-all" }}>
              {t.splitNames[it.split as keyof typeof t.splitNames]} · {it.classes.join(", ")}
              {it.boxes > 0 ? ` · ${t.boxesOnImage(it.boxes)}` : ""}
            </figcaption>
          </figure>
        ))}
      </div>
      {total > 0 && (
        <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
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
      {help && <div style={warn}>{help}</div>}
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

/** What a finished scan found. The sampled notice stays first: a clean sample
 *  proves nothing about the images it skipped. */
export function Report({ report }: { report: ScanReport }) {
  const t = useT().datasetExplorer;
  const { sizes } = report;
  return (
    <>
      <div style={report.sampled || report.cancelled ? warn : text} role="status">
        {report.cancelled
          ? t.scanCancelled(report.scanned)
          : report.sampled
            ? t.scanSampled(report.scanned, report.total_images)
            : t.scanComplete(report.scanned)}
      </div>
      <div className="vf-lab-overview-pair">
        <div style={card}>
          <CardTitle>{t.problemsTitle}</CardTitle>
          {!hasProblems(report) && <div style={text}>{t.noProblems}</div>}
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
            <CardTitle>{t.sizesTitle}</CardTitle>
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
      </div>
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
        <CardTitle>{t.scanTitle}</CardTitle>
        <div style={text}>{t.scanHelp}</div>
        <div className="vf-lab-row">
          <div className="vf-lab-row__field">
            <TextField
              label={t.scanLimit}
              value={limit}
              onChange={setLimit}
              mono
              hint={t.scanLimitHint}
            />
          </div>
          <div className="vf-lab-row__actions vf-lab-row__actions--start">
            {running ? (
              <button type="button" style={ghostButton} onClick={() => void cancel()} disabled={cancelling}>
                {cancelling ? t.scanCancelling : t.scanCancel}
              </button>
            ) : (
              <ActionButton onClick={() => void start()}>{t.scanStart}</ActionButton>
            )}
          </div>
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
        {error && <div style={danger}>{t.scanFailed(error)}</div>}
        {job?.state === "failed" && job.error && <div style={danger}>{t.scanFailed(job.error)}</div>}
      </div>
      {job?.result && <Report report={job.result} />}
    </div>
  );
}

/** "Visão geral": the read-only exploration of the opened dataset (ADR-165).
 *  Totals, the distribution per class and split, a filterable gallery and the
 *  integrity scan with the image sizes. */
export function DatasetOverview({ form, summary }: { form: ExplorerForm; summary: ExplorerSummary }) {
  const t = useT().datasetLab;
  const ex = useT().datasetExplorer;
  return (
    <section className="vf-lab-tool">
      <header className="vf-lab-tool__head">
        <h2 className="vf-lab-tool__title">{t.sections.overview}</h2>
        <p className="vf-lab-tool__lead">{ex.intro}</p>
      </header>
      {summary.total_images === 0 ? (
        <div style={text}>{ex.empty}</div>
      ) : (
        <>
          <TotalCards summary={summary} />
          <Distribution summary={summary} />
          <Gallery form={form} summary={summary} />
          <ScanPanel form={form} />
        </>
      )}
    </section>
  );
}
