import type { Bar, Bin, BoxReport, BoxSize } from "../lib/dataset-explorer";
import { binBars } from "../lib/dataset-explorer";
import { useT } from "../i18n/useT";
import { BarList, DataTable, LearnMore } from "./DatasetLabParts";
import { card, mono, sectionLabel, stepTitle, text, warn } from "./datasetExplorerStyles";

const SIZES: BoxSize[] = ["small", "medium", "large"];
/** Invalid rows listed by path; the full count is above them. */
const EXAMPLES = 20;

/** Whole counts: [2, 4) reads "2–3", [32, ∞) reads "32+". */
function countBars(bins: Bin[]): Bar[] {
  const max = bins.reduce((m, b) => Math.max(m, b.count), 0);
  return bins.map((b) => ({
    label: b.hi === null ? `${b.lo}+` : b.hi - b.lo === 1 ? String(b.lo) : `${b.lo}–${b.hi - 1}`,
    value: b.count,
    pct: max > 0 ? (b.count / max) * 100 : 0,
  }));
}

/** Box centers on the image, darker where more boxes sit. */
function Heatmap({ grid, label }: { grid: number[][]; label: string }) {
  const max = grid.reduce((m, row) => Math.max(m, ...row), 0);
  return (
    <div
      role="img"
      aria-label={label}
      style={{
        display: "grid",
        gridTemplateColumns: `repeat(${grid[0]?.length ?? 1}, 1fr)`,
        gap: 1,
        width: "100%",
        maxWidth: 240,
        aspectRatio: "1",
        border: "1px solid var(--line-vf, rgb(255 255 255 / 0.12))",
      }}
    >
      {grid.flatMap((row, r) =>
        row.map((n, c) => (
          <span
            key={`${r}-${c}`}
            title={String(n)}
            style={{
              background: "var(--accent-vf)",
              opacity: max > 0 ? 0.06 + 0.94 * (n / max) : 0.06,
            }}
          />
        )),
      )}
    </div>
  );
}

/** What the boxes of a detection dataset look like (ADR-233). */
export function BoxStatsCard({ boxes }: { boxes: BoxReport }) {
  const t = useT().datasetExplorer;
  const [small, large] = boxes.size_thresholds;
  const side = (area: number) => Math.round(Math.sqrt(area));
  const sizeLabel = {
    small: t.boxSmall(side(small)),
    medium: t.boxMedium(side(small), side(large)),
    large: t.boxLarge(side(large)),
  };
  const sizeMax = Math.max(0, ...SIZES.map((s) => boxes.sizes[s]));
  // Small, medium, large in that order, not by count.
  const sizeBars = SIZES.map((s) => ({
    label: sizeLabel[s],
    value: boxes.sizes[s],
    pct: sizeMax > 0 ? (boxes.sizes[s] / sizeMax) * 100 : 0,
  }));
  const empty = boxes.no_label_file + boxes.empty_label;
  const invalid = Object.entries(boxes.invalid) as [keyof typeof t.boxProblems, number][];
  return (
    <div style={card} data-testid="box-stats">
      <h3 style={{ ...stepTitle, margin: 0 }}>{t.boxesTitle}</h3>
      <div style={mono}>
        {t.boxTotals(boxes.boxes, boxes.images, boxes.mean_per_image, boxes.max_per_image)}
      </div>
      {empty > 0 && (
        <div style={warn} role="status">
          {t.boxEmpty(empty, boxes.no_label_file, boxes.empty_label)}
        </div>
      )}
      {boxes.invalid_count > 0 && (
        <div style={warn} role="status">
          {t.boxInvalid(boxes.invalid_count, invalid.map(([k, n]) => `${t.boxProblems[k]} (${n})`).join(", "))}
        </div>
      )}
      <div className="vf-lab-overview-pair">
        <div style={{ display: "flex", flexDirection: "column", gap: 8, minWidth: 0 }}>
          <div style={sectionLabel}>{t.boxSizesTitle}</div>
          <BarList bars={sizeBars} />
          <div style={text}>{t.boxSizesHelp}</div>
          {boxes.unsized > 0 && <div style={text}>{t.boxUnsized(boxes.unsized)}</div>}
          <div style={sectionLabel}>{t.boxPerImage}</div>
          <BarList bars={countBars(boxes.per_image_bins)} />
          <div style={sectionLabel}>{t.boxAspect}</div>
          <BarList bars={binBars(boxes.aspect_bins, 2)} />
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 8, minWidth: 0 }}>
          <div style={sectionLabel}>{t.boxHeatmapTitle}</div>
          <Heatmap grid={boxes.heatmap} label={t.boxHeatmapTitle} />
          <div style={text}>{t.boxHeatmapHelp}</div>
          {boxes.pairs.length > 0 && (
            <>
              <div style={sectionLabel}>{t.boxPairsTitle}</div>
              <div style={mono}>
                {boxes.pairs.map((p) => (
                  <div key={`${p.a}+${p.b}`}>{t.boxPair(p.a, p.b, p.images)}</div>
                ))}
              </div>
            </>
          )}
        </div>
      </div>
      {boxes.class_sizes.length > 0 && (
        <LearnMore summary={t.boxClassTable}>
          <DataTable>
            <thead>
              <tr>
                <th style={{ textAlign: "left" }}>{t.classColumn}</th>
                {SIZES.map((s) => (
                  <th key={s} style={{ textAlign: "right" }}>
                    {sizeLabel[s]}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {boxes.class_sizes.map((c) => (
                <tr key={c.name}>
                  <td>{c.name}</td>
                  {SIZES.map((s) => (
                    <td key={s} style={{ textAlign: "right" }}>
                      {c[s]}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </DataTable>
        </LearnMore>
      )}
      {boxes.invalid_examples.length > 0 && (
        <LearnMore summary={t.boxInvalidList}>
          <div style={mono}>
            {boxes.invalid_examples.slice(0, EXAMPLES).map((e) => (
              <div key={`${e.path}:${e.line}`} style={{ wordBreak: "break-all" }}>
                [{t.splitNames[e.split as keyof typeof t.splitNames]}] {e.path} · {t.boxLine(e.line)} ·{" "}
                {t.boxProblems[e.reason]}
              </div>
            ))}
            {boxes.invalid_count > Math.min(EXAMPLES, boxes.invalid_examples.length) && (
              <div>{t.moreNotShown(boxes.invalid_count - Math.min(EXAMPLES, boxes.invalid_examples.length))}</div>
            )}
          </div>
        </LearnMore>
      )}
    </div>
  );
}
