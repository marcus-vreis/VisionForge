import { useEffect, useMemo, useState, type CSSProperties } from "react";
import { fetchCodeTour, type CodeNote, type CodeSection, type CodeSettings } from "../api/client";
import { useT } from "../i18n/useT";
import { loadTourTasks, tourStatus, unavailableReason } from "../lib/code-tour";

const CODE: CSSProperties = {
  margin: 0, padding: "10px 0", overflowX: "auto", maxHeight: 360, overflowY: "auto",
  background: "var(--vf-panel)", border: "1px solid var(--vf-panel-stroke)", borderRadius: 8,
  fontFamily: "var(--font-mono)", fontSize: 12, lineHeight: 1.55,
};

/** What came back for one request: its sections, the server's reason it has none, or a failure. */
interface Loaded {
  request: string;
  sections?: CodeSection[];
  unavailable?: string;
  failed?: boolean;
}

/**
 * The "Código" tab (ADR-173): the real functions the training runs, with notes.
 *
 * The server says which tasks have a tour and reads the code from the
 * installed package, so it is the code that trains; the marked notes are the
 * branch this run's settings take. A run whose code is not this project's
 * gets the server's reason instead.
 */
export function CodeTour({ taskKey, settings, accent }: {
  taskKey: string; settings: CodeSettings | null; accent: string;
}) {
  const t = useT();
  const [tasks, setTasks] = useState<string[] | null>(null);
  const [listFailed, setListFailed] = useState(false);
  const [loaded, setLoaded] = useState<Loaded | null>(null);
  const [open, setOpen] = useState("train");
  const [focus, setFocus] = useState<number | null>(null);
  const query = useMemo(() => JSON.stringify(settings ?? {}), [settings]);
  const status = tourStatus(taskKey, tasks);
  const request = `${taskKey}?${query}`;

  useEffect(() => {
    let alive = true;
    loadTourTasks()
      .then((list) => { if (alive) setTasks(list); })
      .catch(() => { if (alive) setListFailed(true); });
    return () => { alive = false; };
  }, []);

  useEffect(() => {
    if (status !== "supported") return;
    let alive = true;
    const key = `${taskKey}?${query}`;
    fetchCodeTour(taskKey, JSON.parse(query) as CodeSettings)
      .then((result) => { if (alive) setLoaded({ request: key, sections: result }); })
      .catch((error: unknown) => {
        if (!alive) return;
        const reason = unavailableReason(error);
        setLoaded(reason ? { request: key, unavailable: reason } : { request: key, failed: true });
      });
    return () => { alive = false; };
  }, [status, taskKey, query]);

  const current = loaded?.request === request ? loaded : null;
  if (listFailed || current?.failed) return <p role="alert" style={{ fontSize: 13 }}>{t.codeTour.failed}</p>;
  if (status === "missing") return <p style={{ fontSize: 13 }}>{t.codeTour.missing}</p>;
  if (current?.unavailable) return <p style={{ fontSize: 13 }}>{current.unavailable}</p>;
  const sections = current?.sections;
  if (!sections) return <p role="status" style={{ fontSize: 13 }}>{t.codeTour.loading}</p>;
  const section = sections.find((s) => s.id === open) ?? sections[0];

  return <div style={{ display: "flex", flexDirection: "column", gap: 10, fontSize: 13 }}>
    <p style={{ margin: 0, color: "var(--vf-text-dim)" }}>{t.codeTour.intro}</p>
    <div role="tablist" aria-label={t.codeTour.sectionsLabel} style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
      {sections.map((s) => {
        const selected = s.id === section.id;
        return <button key={s.id} type="button" role="tab" aria-selected={selected}
          onClick={() => { setOpen(s.id); setFocus(null); }}
          style={{
            padding: "4px 10px", borderRadius: 999, cursor: "pointer", fontSize: 12,
            border: `1px solid ${selected ? accent : "var(--vf-panel-stroke)"}`,
            background: selected ? "var(--accent-soft)" : "transparent", color: "var(--vf-text)",
          }}>{sectionTitle(s.id, t.codeTour.sections)}</button>;
      })}
    </div>
    <div style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--vf-text-muted)", overflowWrap: "anywhere" }}>
      {t.codeTour.where(section.file, section.name, section.first_line)}
    </div>
    <div style={{ display: "flex", gap: 12, flexWrap: "wrap", alignItems: "flex-start" }}>
      <pre aria-label={section.name} style={{ ...CODE, flex: "2 1 420px", minWidth: 0 }}>
        {section.lines.map((line, index) => {
          const note = noteAt(section.notes, index, focus);
          return <div key={index} style={lineStyle(note, focus !== null && note === section.notes[focus], accent)}>
            <span aria-hidden="true" style={{ display: "inline-block", width: 44, paddingRight: 10, textAlign: "right", color: "var(--vf-text-muted)", userSelect: "none" }}>
              {section.first_line + index}
            </span>
            {line || " "}
          </div>;
        })}
      </pre>
      <ol style={{ flex: "1 1 240px", minWidth: 0, margin: 0, paddingLeft: 18, display: "flex", flexDirection: "column", gap: 8 }}>
        {section.notes.map((note, index) => <li key={index}
          onMouseEnter={() => setFocus(index)} onFocus={() => setFocus(index)}
          onMouseLeave={() => setFocus(null)} tabIndex={0}
          style={{ opacity: note.active === false ? 0.55 : 1 }}>
          <div style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--vf-text-muted)" }}>
            {t.codeTour.lines(section.first_line + note.first, section.first_line + note.last)}
            {note.active === true && <strong style={{ color: accent }}> · {t.codeTour.active}</strong>}
            {note.active === false && <span> · {t.codeTour.inactive}</span>}
          </div>
          <div>{note.text}</div>
        </li>)}
      </ol>
    </div>
    <p style={{ margin: 0, fontSize: 12, color: "var(--vf-text-muted)" }}>{t.codeTour.perEpoch}</p>
  </div>;
}

function sectionTitle(id: string, titles: Record<string, string>): string {
  return titles[id] ?? id;
}

/** The note shown on line `index`: the focused one first, then the first that covers it. */
function noteAt(notes: CodeNote[], index: number, focus: number | null): CodeNote | null {
  const covers = (note: CodeNote) => note.first <= index && index <= note.last;
  if (focus !== null && notes[focus] && covers(notes[focus])) return notes[focus];
  return notes.find(covers) ?? null;
}

function lineStyle(note: CodeNote | null, focused: boolean, accent: string): CSSProperties {
  const base: CSSProperties = { paddingRight: 12, whiteSpace: "pre" };
  if (!note) return base;
  if (note.active === false) return { ...base, opacity: 0.45 };
  return {
    ...base,
    background: focused || note.active ? "var(--accent-soft)" : "rgb(var(--lt-ink, 255 255 255) / 0.04)",
    boxShadow: `inset 3px 0 0 ${note.active || focused ? accent : "var(--vf-panel-stroke)"}`,
  };
}
