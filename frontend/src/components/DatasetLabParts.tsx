import type { ReactNode } from "react";
import { Check, Search } from "lucide-react";
import { useT } from "../i18n/useT";
import type { Bar } from "../lib/dataset-explorer";
import { ActionButton } from "./ui/ActionButton";
import { mono, stepTitle, text } from "./datasetExplorerStyles";

/** The empty part of a bar. */
export const track = "rgb(var(--lt-ink, 255 255 255) / 0.06)";

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

/** The frame every lab tool shares: its name, one sentence on what it does and
 *  what it creates, the long explanation closed behind "saiba mais", then the
 *  tool itself. */
export function LabTool({
  title,
  lead,
  more,
  children,
}: {
  title: string;
  lead: string;
  /** The long explanation, shown only on demand. */
  more?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="vf-lab-tool">
      <header className="vf-lab-tool__head">
        <h2 className="vf-lab-tool__title">{title}</h2>
        <p className="vf-lab-tool__lead">{lead}</p>
        {more && <LearnMore>{more}</LearnMore>}
      </header>
      {children}
    </section>
  );
}

/** A closed disclosure with the long explanation. */
export function LearnMore({ summary, children }: { summary?: string; children: ReactNode }) {
  const t = useT().datasetLab;
  return (
    <details className="vf-lab-more">
      <summary>{summary ?? t.learnMore}</summary>
      <div style={text}>{children}</div>
    </details>
  );
}

/** One numbered step of a tool. */
export function LabStep({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="vf-lab-step">
      <h3 style={{ ...stepTitle, margin: 0 }}>{title}</h3>
      {children}
    </div>
  );
}

/** A form row: the fields grow, the actions sit at its right end (and below
 *  them on a narrow screen). Every tool puts its actions in this place. */
export function ActionRow({ children, actions }: { children?: ReactNode; actions: ReactNode }) {
  return (
    <div className="vf-lab-row">
      {children && <div className="vf-lab-row__field">{children}</div>}
      <div className="vf-lab-row__actions">{actions}</div>
    </div>
  );
}

/** "Analisar": reads the input and fills the preview. */
export function AnalyzeButton({
  label,
  disabled,
  onClick,
}: {
  label: string;
  disabled?: boolean;
  onClick: () => void;
}) {
  return (
    <ActionButton icon={<Search />} disabled={disabled} onClick={onClick}>
      {label}
    </ActionButton>
  );
}

/** The tool's primary action ("Criar dataset novo"), always at the end of the
 *  destination row. */
export function ApplyButton({
  label,
  disabled,
  onClick,
}: {
  label: string;
  disabled?: boolean;
  onClick: () => void;
}) {
  return (
    <ActionButton
      icon={<Check />}
      className="vf-action-btn--primary"
      disabled={disabled}
      onClick={onClick}
    >
      {label}
    </ActionButton>
  );
}

/** A table that scrolls sideways inside its box instead of widening the page. */
export function DataTable({ children, maxWidth }: { children: ReactNode; maxWidth?: number }) {
  return (
    <div className="vf-lab-table" style={maxWidth ? { maxWidth } : undefined}>
      <table style={{ ...mono, borderCollapse: "collapse", width: "100%" }}>{children}</table>
    </div>
  );
}
