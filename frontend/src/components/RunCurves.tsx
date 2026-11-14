/** The run detail's per-epoch charts: train against validation, one chart per
 *  quantity the task leads with (lib/run-curves.ts). Hovering reads the exact
 *  value of each line at that epoch. */

import type { RunDetail } from "../api/client";
import { useT } from "../i18n/useT";
import { formatCurveValue } from "../lib/compare-curves";
import { runTaskKey } from "../lib/compare-metrics";
import { TRAIN_COLOR, VAL_COLOR, curvePoints, runCurveGroups } from "../lib/run-curves";
import { LineChart } from "./ui/LineChart";

export function RunCurves({ detail }: { detail: RunDetail }) {
  const t = useT();
  const groups = runCurveGroups(runTaskKey(detail), detail.history, detail.metric_directions);
  if (groups.length === 0) return null;

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 360px), 1fr))",
        gap: 12,
      }}
    >
      {groups.map((group) => {
        const series = group.series.map((s) => ({
          key: s.key,
          name: s.label ? t.compareRuns.curves[s.label] : s.key,
          color: s.role === "train" ? TRAIN_COLOR : VAL_COLOR,
          points: curvePoints(detail.history, s.key),
        }));
        const title = series.map((s) => s.name).join(" · ");
        return (
          <figure
            key={group.id}
            style={{
              margin: 0,
              minWidth: 0,
              padding: "12px 12px 8px",
              background: "rgb(var(--lt-shade, 0 0 0) / 0.3)",
              border: "1px solid var(--vf-panel-stroke)",
              borderRadius: 10,
            }}
          >
            <figcaption
              style={{
                display: "flex",
                flexWrap: "wrap",
                gap: 14,
                marginBottom: 8,
                fontSize: 13,
                color: "var(--vf-text-dim)",
              }}
            >
              {series.map((s) => (
                <span key={s.key} style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                  <span
                    aria-hidden
                    style={{ width: 10, height: 10, borderRadius: 3, background: s.color }}
                  />
                  {s.name}
                </span>
              ))}
            </figcaption>
            <LineChart title={title} height={190} width={420} formatValue={formatCurveValue} series={series} />
          </figure>
        );
      })}
    </div>
  );
}
