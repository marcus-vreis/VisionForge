import { useId, type CSSProperties, type ReactNode } from "react";
import { FlaskConical, Play, Repeat, RotateCcw } from "lucide-react";
import { useT } from "../i18n/useT";
import {
  nextStepItems,
  type NextStep,
  type NextStepAction,
  type NextStepHandlers,
} from "../lib/next-steps";
import { ActionButton } from "./ui/ActionButton";

const ICONS: Record<NextStepAction, ReactNode> = {
  resume: <Play />,
  reuse: <RotateCcw />,
  test: <FlaskConical />,
  replicate: <Repeat />,
};

const BOX_STYLE: CSSProperties = {
  padding: "14px 16px",
  background: "rgb(var(--lt-ink, 255 255 255) / 0.02)",
  border: "1px solid var(--vf-panel-stroke)",
  borderRadius: 12,
};

const TITLE_STYLE: CSSProperties = {
  margin: "0 0 10px",
  fontFamily: "var(--font-mono)",
  fontSize: 11,
  fontWeight: 500,
  letterSpacing: "0.16em",
  textTransform: "uppercase",
  color: "var(--vf-text-muted)",
};

const ITEM_STYLE: CSSProperties = {
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
  flexWrap: "wrap",
  gap: "8px 12px",
};

const TEXT_STYLE: CSSProperties = {
  flex: "1 1 220px",
  minWidth: 0,
  fontSize: 13,
  lineHeight: 1.5,
  color: "var(--vf-text-dim)",
  overflowWrap: "anywhere",
};

/** "E agora?": one or two short suggestions under a run's numbers; nothing when none holds. */
export function NextSteps({
  steps,
  handlers,
  disabled,
}: {
  steps: readonly NextStep[];
  handlers: NextStepHandlers;
  disabled?: Partial<Record<NextStepAction, boolean>>;
}) {
  const t = useT();
  const headingId = useId();
  const items = nextStepItems(t, steps, handlers, disabled);
  if (items.length === 0) return null;
  return (
    <section aria-labelledby={headingId} style={BOX_STYLE}>
      <h3 id={headingId} style={TITLE_STYLE}>
        {t.nextSteps.title}
      </h3>
      <ul style={{ listStyle: "none", margin: 0, padding: 0, display: "flex", flexDirection: "column", gap: 10 }}>
        {items.map((item) => (
          <li key={item.id} style={ITEM_STYLE}>
            <span style={TEXT_STYLE}>{item.text}</span>
            {item.action && (
              <ActionButton
                icon={ICONS[item.action.kind]}
                onClick={item.action.onClick}
                disabled={item.action.disabled}
                style={{ maxWidth: "100%" }}
              >
                {item.action.label}
              </ActionButton>
            )}
          </li>
        ))}
      </ul>
    </section>
  );
}
