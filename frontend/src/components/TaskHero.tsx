import type { TaskDefinition } from "../types/tasks";

interface DataPillProps {
  label: string;
  value: string;
}

function DataPill({ label, value }: DataPillProps) {
  return (
    <div
      style={{
        padding: "8px 14px",
        borderRadius: 10,
        background: "rgba(8,10,14,0.55)",
        border: "1px solid var(--vf-panel-stroke)",
      }}
    >
      <div
        style={{
          fontSize: 9.5,
          letterSpacing: "0.20em",
          textTransform: "uppercase",
          color: "var(--vf-text-muted)",
          fontFamily: "var(--font-mono)",
        }}
      >
        {label}
      </div>
      <div
        style={{
          fontSize: 12,
          color: "var(--vf-text)",
          marginTop: 2,
          fontFamily: "var(--font-mono)",
        }}
      >
        {value}
      </div>
    </div>
  );
}

interface TaskHeroProps {
  task: TaskDefinition;
  baseDir?: string;
}

/** Hero section with task title, description, and dataset pills. */
export function TaskHero({ task, baseDir }: TaskHeroProps) {
  const words = task.label.split(" ");
  return (
    <div
      key={task.key}
      style={{
        display: "flex",
        alignItems: "flex-end",
        justifyContent: "space-between",
        marginBottom: 26,
        animation: "fadeUp 320ms ease forwards",
        flexWrap: "wrap",
        gap: 16,
      }}
    >
      <div>
        <div
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: 11,
            color: "var(--vf-text-muted)",
            letterSpacing: "0.22em",
            textTransform: "uppercase",
          }}
        >
          // task / {task.short}
        </div>
        <h1
          style={{
            fontSize: 44,
            fontWeight: 600,
            letterSpacing: "-0.025em",
            margin: "8px 0 6px",
            lineHeight: 1.05,
            fontFamily: "var(--font-sans)",
          }}
        >
          {words.map((w, i) =>
            i === words.length - 1 ? (
              <span
                key={i}
                style={{
                  background:
                    "linear-gradient(180deg, var(--vf-text) 0%, var(--vf-text-dim) 100%)",
                  WebkitBackgroundClip: "text",
                  backgroundClip: "text",
                  color: "transparent",
                }}
              >
                {w}
              </span>
            ) : (
              <span key={i}>{w} </span>
            ),
          )}
        </h1>
        <p
          style={{
            color: "var(--vf-text-dim)",
            fontSize: 14,
            maxWidth: 540,
            lineHeight: 1.5,
            margin: 0,
          }}
        >
          {task.description}.
        </p>
      </div>

      <div
        style={{
          display: "flex",
          gap: 10,
          fontFamily: "var(--font-mono)",
          fontSize: 11,
          color: "var(--vf-text-muted)",
          letterSpacing: "0.08em",
          flexWrap: "wrap",
        }}
      >
        {baseDir && <DataPill label="Dataset" value={baseDir} />}
        <DataPill label="Task" value={task.short} />
      </div>
    </div>
  );
}
