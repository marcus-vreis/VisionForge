import type { SplitBalance } from "../lib/dataset-explorer";
import { useT } from "../i18n/useT";
import { DataTable, LearnMore } from "./DatasetLabParts";
import { card, mono, stepTitle, text, warn } from "./datasetExplorerStyles";

/** Classes named under the verdict: the largest gaps are the ones that matter. */
const TOP = 5;

function pct(value: number | undefined): string {
  return value === undefined ? "—" : `${value.toFixed(1)}%`;
}

/** Whether the splits share the class mix (ADR-232). */
export function SplitBalanceCard({ balance }: { balance: SplitBalance }) {
  const t = useT().datasetLab;
  const ex = useT().datasetExplorer;
  if (balance.verdict === "one_split") return null;
  const [stratified, close] = balance.thresholds_pp;
  const good = balance.verdict === "stratified";
  const missing = balance.classes.filter((c) => c.missing_in.length > 0);
  const names = (splits: string[]) =>
    splits.map((s) => ex.splitNames[s as keyof typeof ex.splitNames]).join(", ");
  return (
    <div style={card} data-testid="split-balance">
      <div style={stepTitle}>{t.balanceTitle}</div>
      <div style={good ? text : warn} role="status">
        {t.balanceVerdict[balance.verdict](balance.max_gap_pp, stratified, close)}
      </div>
      {missing.length > 0 && (
        <div style={warn}>{missing.map((c) => t.balanceMissing(c.name, names(c.missing_in))).join(" ")}</div>
      )}
      {!good && balance.classes.length > 0 && (
        <div style={text}>
          {t.balanceTop(
            balance.classes
              .slice(0, TOP)
              .map((c) => `${c.name} (${c.gap_pp.toFixed(1)})`)
              .join(", "),
          )}
        </div>
      )}
      <div style={mono}>
        {balance.cramers_v !== null && <div>{t.balanceCramer(balance.cramers_v)}</div>}
        <div>
          {balance.splits
            .map((s) => t.balanceImbalance(ex.splitNames[s], balance.imbalance[s] ?? null))
            .join("  ·  ")}
        </div>
      </div>
      <LearnMore summary={t.balanceTable}>
        <DataTable>
          <thead>
            <tr>
              <th style={{ textAlign: "left" }}>{ex.classColumn}</th>
              {balance.splits.map((s) => (
                <th key={s} style={{ textAlign: "right" }}>
                  {ex.splitNames[s]}
                </th>
              ))}
              <th style={{ textAlign: "right" }}>{t.balanceGap}</th>
            </tr>
          </thead>
          <tbody>
            {balance.classes.map((c) => (
              <tr key={c.name}>
                <td>{c.name}</td>
                {balance.splits.map((s) => (
                  <td key={s} style={{ textAlign: "right" }}>
                    {pct(c.shares[s])}
                  </td>
                ))}
                <td style={{ textAlign: "right" }}>{c.gap_pp.toFixed(1)}</td>
              </tr>
            ))}
          </tbody>
        </DataTable>
      </LearnMore>
      <div style={text}>{t.balanceHelp}</div>
    </div>
  );
}
