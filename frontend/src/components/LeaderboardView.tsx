import { useT } from "../i18n/useT";
import type { MetricLabelKey } from "../lib/compare-metrics";
import { shortDigest } from "../lib/dataset-identity";
import {
  boardMetrics,
  defaultMetric,
  groupBoards,
  rankEntries,
  type Board,
  type RankedRow,
  type UnrankedRow,
} from "../lib/leaderboard";
import { formatAggregate, formatNumber, type HistoryEntry } from "../lib/run-groups";
import { accentForTask } from "../lib/task-accent";
import { familyLabel, taskFamily } from "../lib/task-family";
import { MenuSelect } from "./controls";

/** The amber the History already uses for a run a stop cut, here for a caution. */
const CAUTION_BG = "oklch(0.84 0.12 85 / 0.08)";
const CAUTION_BORDER = "oklch(0.84 0.12 85 / 0.4)";
const CAUTION_TEXT = "oklch(0.90 0.12 85)";

interface LeaderboardViewProps {
  /** History's entries, a replicate group being one with its seeds folded under it. */
  entries: HistoryEntry[];
  /** The task tab History is on: "all" or a family. */
  taskFilter: string;
  /** Run ids ticked for a comparison, across boards (each board reads its own). */
  ticks: string[];
  onToggleTick: (runId: string) => void;
  onCompare: (runIds: string[]) => void;
  onOpenRun: (runId: string) => void;
  /** The metric picked per board id; a board without one opens on its default. */
  metricPicks: Record<string, string>;
  onPickMetric: (boardId: string, metric: string) => void;
}

const MONO = "var(--font-mono)";

function Pill({
  children,
  color,
  title,
}: {
  children: React.ReactNode;
  color: string;
  title?: string;
}) {
  return (
    <span
      title={title}
      style={{
        padding: "2px 8px",
        background: `color-mix(in oklch, ${color} 12%, transparent)`,
        border: `1px solid color-mix(in oklch, ${color} 35%, transparent)`,
        borderRadius: 999,
        fontFamily: MONO,
        fontSize: 10,
        color,
        letterSpacing: "0.10em",
        textTransform: "uppercase",
        whiteSpace: "nowrap",
      }}
    >
      {children}
    </span>
  );
}

function TickBox({
  checked,
  onToggle,
}: {
  checked: boolean;
  onToggle: () => void;
}) {
  const t = useT();
  return (
    <button
      type="button"
      role="checkbox"
      aria-checked={checked}
      aria-label={t.leaderboard.tickTitle}
      title={t.leaderboard.tickTitle}
      onClick={onToggle}
      style={{
        width: 18,
        height: 18,
        padding: 0,
        borderRadius: 4,
        border: `1.5px solid ${checked ? "oklch(0.78 0.18 150)" : "var(--vf-panel-stroke)"}`,
        background: checked ? "oklch(0.78 0.18 150 / 0.3)" : "transparent",
        color: "var(--vf-text)",
        fontSize: 11,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        cursor: "pointer",
        flexShrink: 0,
      }}
    >
      {checked ? "✓" : ""}
    </button>
  );
}

/** The name of a run, which opens it, and what it ran. */
function RunName({
  entry,
  onOpen,
  tag,
  accent,
}: {
  entry: HistoryEntry;
  onOpen: () => void;
  tag?: string;
  accent: string;
}) {
  const t = useT();
  const { run } = entry;
  return (
    <div style={{ minWidth: 0, display: "flex", flexDirection: "column", gap: 3 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, minWidth: 0 }}>
        <button
          type="button"
          onClick={onOpen}
          title={t.leaderboard.openTitle}
          style={{
            padding: 0,
            background: "transparent",
            border: "none",
            color: "var(--vf-text)",
            fontWeight: 600,
            fontSize: 13,
            textAlign: "left",
            cursor: "pointer",
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
            minWidth: 0,
          }}
        >
          {run.experiment_name}
        </button>
        {tag && <Pill color={accent}>{tag}</Pill>}
      </div>
      <div
        style={{
          fontFamily: MONO,
          fontSize: 11,
          color: "var(--vf-text-muted)",
          overflow: "hidden",
          textOverflow: "ellipsis",
          whiteSpace: "nowrap",
        }}
      >
        {run.model_arch}
      </div>
    </div>
  );
}

function RankedLine({
  row,
  accent,
  ticked,
  onToggleTick,
  onOpen,
}: {
  row: RankedRow;
  accent: string;
  ticked: boolean;
  onToggleTick: () => void;
  onOpen: () => void;
}) {
  const t = useT();
  const shown = row.seeds >= 2 ? formatAggregate(row.aggregate) : null;
  const main = shown?.mean ?? formatNumber(row.value);
  const mean = row.basis === "mean";
  return (
    <div style={lineStyle(ticked)}>
      <TickBox checked={ticked} onToggle={onToggleTick} />
      <span
        title={t.leaderboard.topTag(mean)}
        style={{ fontFamily: MONO, fontSize: 13, color: accent, textAlign: "right" }}
      >
        {t.leaderboard.rank(row.rank)}
      </span>
      <RunName
        entry={row.entry}
        onOpen={onOpen}
        accent={accent}
        tag={row.rank === 1 ? t.leaderboard.topTag(mean) : undefined}
      />
      <div
        title={
          row.low !== null && row.high !== null
            ? t.leaderboard.ciTitle(formatNumber(row.low), formatNumber(row.high))
            : undefined
        }
        style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 2 }}
      >
        <span style={{ fontFamily: MONO, fontSize: 13, fontWeight: 600, color: accent }}>
          {main}
          {shown?.half != null && (
            <span style={{ fontSize: 11, fontWeight: 400, color: "var(--vf-text-muted)" }}>
              {" ± "}
              {shown.half}
            </span>
          )}
        </span>
        <span style={{ fontFamily: MONO, fontSize: 10, color: "var(--vf-text-muted)" }}>
          {mean ? t.leaderboard.valueNote.mean(row.seeds) : t.leaderboard.valueNote.value}
        </span>
      </div>
    </div>
  );
}

function UnrankedLine({
  row,
  accent,
  ticked,
  onToggleTick,
  onOpen,
}: {
  row: UnrankedRow;
  accent: string;
  ticked?: boolean;
  onToggleTick?: () => void;
  onOpen: () => void;
}) {
  const t = useT();
  return (
    <div style={lineStyle(ticked === true)}>
      {onToggleTick ? <TickBox checked={ticked === true} onToggle={onToggleTick} /> : <span />}
      <span style={{ fontFamily: MONO, fontSize: 13, color: "var(--vf-text-muted)", textAlign: "right" }}>
        —
      </span>
      <RunName entry={row.entry} onOpen={onOpen} accent={accent} />
      <span
        style={{
          fontFamily: MONO,
          fontSize: 11,
          color: "var(--vf-text-muted)",
          textAlign: "right",
          maxWidth: 220,
        }}
      >
        {t.leaderboard.reason[row.reason]}
      </span>
    </div>
  );
}

function lineStyle(ticked: boolean): React.CSSProperties {
  return {
    display: "grid",
    gridTemplateColumns: "20px 34px minmax(0, 1fr) auto",
    alignItems: "center",
    gap: 10,
    padding: "8px 10px",
    borderRadius: 8,
    background: ticked ? "rgba(120, 200, 130, 0.10)" : "rgba(255,255,255,0.02)",
    border: `1px solid ${ticked ? "oklch(0.78 0.18 150 / 0.6)" : "transparent"}`,
  };
}

function BoardCard({
  board,
  metricKey,
  onPickMetric,
  ticks,
  onToggleTick,
  onCompare,
  onOpenRun,
}: {
  board: Board;
  metricKey: string | undefined;
  onPickMetric: (metric: string) => void;
  ticks: string[];
  onToggleTick: (runId: string) => void;
  onCompare: (runIds: string[]) => void;
  onOpenRun: (runId: string) => void;
}) {
  const t = useT();
  const unknown = board.identity.basis === "unknown";
  const accent = accentForTask(board.task);
  const family = taskFamily(board.task);

  const metrics = unknown ? [] : boardMetrics(board.entries);
  const active =
    metrics.find((m) => m.key === metricKey) ?? metrics.find((m) => m.key === defaultMetric(metrics));
  const ranking = rankEntries(board.entries, active?.key ?? "", active?.direction ?? "higher");
  const labelOf = (m: { key: string; label: MetricLabelKey | null }) =>
    m.label ? t.compareRuns.metrics[m.label] : m.key;

  const boardIds = new Set(board.entries.map((e) => e.run.run_id));
  const boardTicks = ticks.filter((id) => boardIds.has(id));

  return (
    <section
      style={{
        padding: "14px 16px",
        background: "rgba(255,255,255,0.025)",
        border: "1px solid var(--vf-panel-stroke)",
        borderRadius: 12,
        display: "flex",
        flexDirection: "column",
        gap: 10,
      }}
    >
      <header style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
          <Pill color={accent}>{familyLabel(t, family)}</Pill>
          {board.task !== family && !board.task.startsWith("custom:") && (
            <Pill color={accent}>{board.task}</Pill>
          )}
          <span style={{ fontWeight: 600, fontSize: 14, color: "var(--vf-text)" }}>
            {unknown ? t.leaderboard.unknownTitle : (board.name ?? t.leaderboard.unnamedDataset)}
          </span>
          {!unknown && (
            <Pill
              color="oklch(0.86 0.11 250)"
              title={
                board.identity.basis === "fingerprint"
                  ? t.leaderboard.basisTitle.fingerprint(
                      board.method ?? "",
                      shortDigest(board.digest),
                    )
                  : t.leaderboard.basisTitle.path
              }
            >
              {board.identity.basis === "fingerprint"
                ? t.leaderboard.basis.fingerprint
                : t.leaderboard.basis.path}
            </Pill>
          )}
          <span
            style={{ marginLeft: "auto", fontFamily: MONO, fontSize: 11, color: "var(--vf-text-muted)" }}
          >
            {t.leaderboard.runs(board.entries.length)}
          </span>
        </div>
        {unknown ? (
          <div style={{ fontFamily: MONO, fontSize: 11, color: "var(--vf-text-muted)", lineHeight: 1.55 }}>
            {t.leaderboard.unknownBody}
          </div>
        ) : (
          board.root && (
            <div
              title={board.root}
              style={{
                fontFamily: MONO,
                fontSize: 10,
                color: "var(--vf-text-muted)",
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
              }}
            >
              {board.root}
            </div>
          )
        )}
      </header>

      {active && (
        <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
          <span
            style={{
              fontFamily: MONO,
              fontSize: 9,
              letterSpacing: "0.16em",
              textTransform: "uppercase",
              color: "var(--vf-text-muted)",
            }}
          >
            {t.leaderboard.metricLabel}
          </span>
          <MenuSelect
            value={active.key}
            onChange={onPickMetric}
            options={metrics.map((m) => ({ value: m.key, label: labelOf(m) }))}
            minWidth={170}
          />
          <span style={{ fontFamily: MONO, fontSize: 11, color: "var(--vf-text-muted)" }}>
            {t.compareRuns.direction[active.direction]}
          </span>
          {boardTicks.length >= 2 && (
            <button
              type="button"
              onClick={() => onCompare(boardTicks)}
              style={{
                marginLeft: "auto",
                padding: "6px 12px",
                background: "oklch(0.78 0.18 150 / 0.30)",
                border: "1px solid oklch(0.78 0.18 150)",
                borderRadius: 10,
                color: "var(--vf-text)",
                fontFamily: MONO,
                fontSize: 11,
                letterSpacing: "0.10em",
                textTransform: "uppercase",
                cursor: "pointer",
                fontWeight: 600,
              }}
            >
              {t.history.compare(boardTicks.length)}
            </button>
          )}
        </div>
      )}

      {active && (
        <div style={{ fontFamily: MONO, fontSize: 11, color: "var(--vf-text-muted)", lineHeight: 1.55 }}>
          {t.leaderboard.legend(labelOf(active), t.compareRuns.direction[active.direction])}
        </div>
      )}

      {ranking.caution && (
        <div
          role="note"
          style={{
            padding: "8px 12px",
            background: CAUTION_BG,
            border: `1px solid ${CAUTION_BORDER}`,
            borderRadius: 8,
            fontFamily: MONO,
            fontSize: 11,
            lineHeight: 1.55,
            color: CAUTION_TEXT,
          }}
        >
          {t.leaderboard.caution[ranking.caution]}
        </div>
      )}

      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        {ranking.ranked.map((row) => (
          <RankedLine
            key={row.entry.run.run_id}
            row={row}
            accent={accent}
            ticked={ticks.includes(row.entry.run.run_id)}
            onToggleTick={() => onToggleTick(row.entry.run.run_id)}
            onOpen={() => onOpenRun(row.entry.run.run_id)}
          />
        ))}
      </div>

      {!unknown && ranking.unranked.length > 0 && (
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <span
            style={{
              fontFamily: MONO,
              fontSize: 9,
              letterSpacing: "0.16em",
              textTransform: "uppercase",
              color: "var(--vf-text-muted)",
              marginTop: 4,
            }}
          >
            {t.leaderboard.unrankedTitle}
          </span>
          {ranking.unranked.map((row) => (
            <UnrankedLine
              key={row.entry.run.run_id}
              row={row}
              accent={accent}
              ticked={ticks.includes(row.entry.run.run_id)}
              onToggleTick={() => onToggleTick(row.entry.run.run_id)}
              onOpen={() => onOpenRun(row.entry.run.run_id)}
            />
          ))}
        </div>
      )}

      {unknown && (
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          {board.entries.map((entry) => (
            <div key={entry.run.run_id} style={lineStyle(false)}>
              <span />
              <span />
              <RunName entry={entry} onOpen={() => onOpenRun(entry.run.run_id)} accent={accent} />
              <span />
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

/**
 * The History as a ranking: one board per task and dataset, each ordered by a
 * metric the researcher can switch. An order of numbers, with the caution that
 * goes with it when the top of it rests on one seed apiece or on intervals that
 * overlap (lib/leaderboard.ts).
 */
export function LeaderboardView({
  entries,
  taskFilter,
  ticks,
  onToggleTick,
  onCompare,
  onOpenRun,
  metricPicks,
  onPickMetric,
}: LeaderboardViewProps) {
  const t = useT();
  const boards = groupBoards(
    entries.filter((e) => taskFilter === "all" || taskFamily(e.run.task) === taskFilter),
  );
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <div style={{ fontFamily: MONO, fontSize: 11, color: "var(--vf-text-muted)", lineHeight: 1.6 }}>
        {t.leaderboard.intro}
      </div>
      {boards.length === 0 ? (
        <div
          style={{
            padding: 24,
            fontFamily: MONO,
            fontSize: 12,
            color: "var(--vf-text-muted)",
            textAlign: "center",
            border: "1px dashed var(--vf-panel-stroke)",
            borderRadius: 10,
          }}
        >
          {t.history.noMatch}
        </div>
      ) : (
        boards.map((board) => (
          <BoardCard
            key={board.id}
            board={board}
            metricKey={metricPicks[board.id]}
            onPickMetric={(metric) => onPickMetric(board.id, metric)}
            ticks={ticks}
            onToggleTick={onToggleTick}
            onCompare={onCompare}
            onOpenRun={onOpenRun}
          />
        ))
      )}
    </div>
  );
}
