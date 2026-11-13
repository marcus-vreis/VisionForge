import { Fragment, useState, type CSSProperties } from "react";
import type { CodeSection } from "../../api/client";
import { useT } from "../../i18n/useT";
import { lineMarks } from "../../lib/code-lines";
import { libraryNotice } from "../../lib/code-tour";

const MONO: CSSProperties = { fontFamily: "var(--font-mono)" };
const GUTTER = 22;
const NUMBER = 44;

/**
 * One real function of the training (ADR-173): its lines as they are in the
 * installed package, numbered as in the file, with the branches this run does
 * not take faded. A line range with a note has a quiet marker in the gutter;
 * the note opens in place on a click, or Enter on the marker.
 */
export function CodeView({ section, accent }: { section: CodeSection; accent: string }) {
  const t = useT();
  const words = t.codeTour;
  // Open notes belong to the function they were opened in.
  const [open, setOpen] = useState<{ id: string; notes: number[] }>({ id: section.id, notes: [] });
  const opened = open.id === section.id ? open.notes : [];
  // One marker per line: notes that start on the same line open and close together.
  const toggle = (notes: number[], isOpen: boolean) =>
    setOpen({
      id: section.id,
      notes: isOpen ? opened.filter((i) => !notes.includes(i)) : [...opened, ...notes],
    });
  const marks = lineMarks(section.notes, section.lines.length);
  const notice = libraryNotice(section, words);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8, minWidth: 0 }}>
      <div style={{ ...MONO, fontSize: 11, color: "var(--vf-text-muted)", overflowWrap: "anywhere" }}>
        {words.where(section.file, section.name, section.first_line)}
      </div>
      <p style={{ margin: 0, fontSize: 13, lineHeight: 1.5, color: "var(--vf-text)" }}>{section.summary}</p>
      {notice && (
        <p role="note" style={{ margin: 0, fontSize: 13, lineHeight: 1.5, color: "var(--vf-text-dim)" }}>{notice}</p>
      )}
      <div
        role="region"
        aria-label={section.name}
        data-code-view={section.id}
        style={{
          ...MONO,
          fontSize: 12.5,
          lineHeight: 1.6,
          padding: "8px 0",
          overflowX: "auto",
          background: "var(--vf-panel)",
          border: "1px solid var(--vf-panel-stroke)",
          borderRadius: 8,
        }}
      >
        {section.lines.map((line, index) => {
          const mark = marks[index];
          return (
            <Fragment key={index}>
              <div style={{ display: "flex", minWidth: "max-content", opacity: mark.faded ? 0.45 : 1 }}>
                <span
                  style={{
                    width: GUTTER,
                    flexShrink: 0,
                    display: "inline-flex",
                    alignItems: "center",
                    justifyContent: "center",
                    boxShadow: mark.noted ? "inset -2px 0 0 var(--vf-panel-stroke)" : undefined,
                  }}
                >
                  {mark.starts.length > 0 && (() => {
                    const first = section.notes[mark.starts[0]];
                    const range = words.lines(section.first_line + first.first, section.first_line + first.last);
                    const isOpen = mark.starts.every((note) => opened.includes(note));
                    return (
                      <button
                        type="button"
                        data-note-marker
                        aria-expanded={isOpen}
                        aria-label={words.noteToggle(range)}
                        title={words.noteToggle(range)}
                        onClick={() => toggle(mark.starts, isOpen)}
                        style={{
                          width: 14,
                          height: 14,
                          padding: 0,
                          borderRadius: "50%",
                          cursor: "pointer",
                          border: `1.5px solid ${isOpen ? accent : "var(--vf-text-muted)"}`,
                          background: isOpen ? accent : "transparent",
                        }}
                      />
                    );
                  })()}
                </span>
                <span
                  aria-hidden="true"
                  style={{
                    width: NUMBER,
                    flexShrink: 0,
                    paddingRight: 10,
                    textAlign: "right",
                    fontSize: 11,
                    color: "var(--vf-text-muted)",
                    userSelect: "none",
                  }}
                >
                  {section.first_line + index}
                </span>
                <span style={{ whiteSpace: "pre", paddingRight: 12, color: "var(--vf-text)" }}>{line || " "}</span>
              </div>
              {mark.ends
                .filter((note) => opened.includes(note))
                .map((note) => {
                  const { text, active } = section.notes[note];
                  return (
                    <div
                      key={`note-${note}`}
                      role="note"
                      style={{
                        position: "sticky",
                        left: 0,
                        boxSizing: "border-box",
                        maxWidth: "100%",
                        margin: `4px 10px 6px ${GUTTER + 6}px`,
                        padding: "6px 10px",
                        borderLeft: `3px solid ${active === false ? "var(--vf-panel-stroke)" : accent}`,
                        background: "rgb(var(--lt-ink, 255 255 255) / 0.04)",
                        borderRadius: 6,
                        fontFamily: "var(--font-sans)",
                        fontSize: 13,
                        lineHeight: 1.5,
                        whiteSpace: "normal",
                        color: "var(--vf-text)",
                      }}
                    >
                      {active !== null && (
                        <span style={{ ...MONO, fontSize: 11, marginRight: 8, color: active ? accent : "var(--vf-text-muted)" }}>
                          {active ? words.active : words.inactive}
                        </span>
                      )}
                      {text}
                    </div>
                  );
                })}
            </Fragment>
          );
        })}
      </div>
    </div>
  );
}
