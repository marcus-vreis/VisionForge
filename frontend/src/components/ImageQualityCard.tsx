import type { QualityKind, QualityMetric, QualityOutlier, QualityReport } from "../lib/dataset-explorer";
import { thumbUrl } from "../lib/dataset-explorer";
import { useT } from "../i18n/useT";
import { DataTable, LearnMore } from "./DatasetLabParts";
import { card, mono, sectionLabel, stepTitle, text, warn } from "./datasetExplorerStyles";

const METRICS: QualityMetric[] = ["brightness", "contrast", "sharpness"];
const KINDS: QualityKind[] = ["dark", "bright", "flat", "blurry"];
// The splits, and the anomaly groups (ADR-240): train first, defect kinds last.
const SPLIT_ORDER = ["train", "val", "test", "test_normal"];
/** Thumbnails per kind; the count above them is the full one. */
const SHOWN = 12;

function fmt(v: number): string {
  return v >= 100 ? v.toFixed(0) : v.toFixed(1);
}

function Outliers({ root, kind, items, total }: { root: string; kind: QualityKind; items: QualityOutlier[]; total: number }) {
  const t = useT().imageQuality;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }} data-testid={`quality-${kind}`}>
      <div style={sectionLabel}>
        {t.kinds[kind]} ({total})
      </div>
      <div style={text}>{t.kindHelp[kind]}</div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(96px, 1fr))", gap: 8 }}>
        {items.slice(0, SHOWN).map((o) => (
          <figure key={`${o.split}/${o.path}`} style={{ margin: 0, minWidth: 0 }}>
            <img
              src={thumbUrl(root, o.path, 128, null)}
              alt={o.path}
              loading="lazy"
              style={{ width: "100%", aspectRatio: "1", objectFit: "cover", borderRadius: 6 }}
            />
            <figcaption style={{ ...mono, fontSize: 11, wordBreak: "break-all" }} title={o.path}>
              {o.path} · {fmt(o.value)}
            </figcaption>
          </figure>
        ))}
      </div>
      {total > Math.min(items.length, SHOWN) && <div style={mono}>{t.more(total - Math.min(items.length, SHOWN))}</div>}
    </div>
  );
}

/** Brightness, contrast and sharpness of the scanned images, and the outliers (ADR-238). */
export function ImageQualityCard({
  quality,
  root,
  splitLabel,
}: {
  quality: QualityReport;
  root: string;
  /** How to name a split or group key; the split names by default. */
  splitLabel?: (key: string) => string;
}) {
  const t = useT().imageQuality;
  const splitNames = useT().datasetExplorer.splitNames;
  if (!quality.metrics || !quality.outlier_count || !quality.outliers) return null;
  const { metrics, outlier_count: counts, outliers } = quality;
  // Train, validation, test; any other split name after them.
  const rank = (s: string) => (SPLIT_ORDER.includes(s) ? SPLIT_ORDER.indexOf(s) : SPLIT_ORDER.length);
  const splits = Object.keys(metrics.brightness.median_by_split).sort((a, b) => rank(a) - rank(b));
  const flagged = KINDS.filter((k) => counts[k] > 0);
  const label = splitLabel ?? ((s: string) => splitNames[s as keyof typeof splitNames] ?? s);
  return (
    <div style={card} data-testid="image-quality">
      <h3 style={{ ...stepTitle, margin: 0 }}>{t.title}</h3>
      <div style={text}>{t.intro(quality.count)}</div>
      <DataTable>
        <thead>
          <tr>
            <th style={{ textAlign: "left" }}>{t.metric}</th>
            <th style={{ textAlign: "right" }}>{t.median}</th>
            <th style={{ textAlign: "right" }}>{t.middleHalf}</th>
            <th style={{ textAlign: "right" }}>{t.normalRange}</th>
            {splits.length > 1 &&
              splits.map((s) => (
                <th key={s} style={{ textAlign: "right" }}>
                  {t.medianIn(label(s))}
                </th>
              ))}
          </tr>
        </thead>
        <tbody>
          {METRICS.map((m) => {
            const r = metrics[m];
            const [lo, hi] = r.fences;
            return (
              <tr key={m}>
                <td>{t.metrics[m]}</td>
                <td style={{ textAlign: "right" }}>{fmt(r.median)}</td>
                <td style={{ textAlign: "right" }}>
                  {fmt(r.q1)}–{fmt(r.q3)}
                </td>
                <td style={{ textAlign: "right" }}>{t.range(m, Math.max(lo, 0), hi)}</td>
                {splits.length > 1 &&
                  splits.map((s) => (
                    <td key={s} style={{ textAlign: "right" }}>
                      {fmt(r.median_by_split[s] ?? 0)}
                    </td>
                  ))}
              </tr>
            );
          })}
        </tbody>
      </DataTable>
      {flagged.length === 0 ? (
        <div style={text}>{t.none}</div>
      ) : (
        <div style={warn} role="status">
          {t.found(flagged.map((k) => `${t.kinds[k].toLowerCase()} (${counts[k]})`).join(", "))}
        </div>
      )}
      {flagged.map((k) => (
        <Outliers key={k} root={root} kind={k} items={outliers[k]} total={counts[k]} />
      ))}
      <LearnMore summary={t.howTitle}>
        <div style={text}>{t.how(quality.measure_side ?? 256, quality.tukey_k ?? 1.5)}</div>
      </LearnMore>
    </div>
  );
}
