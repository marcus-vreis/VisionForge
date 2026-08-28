import type { TaskDefinition } from "../types/tasks";

interface TabBarProps {
  tasks: TaskDefinition[];
  activeKey: string;
  setActiveKey: (key: string) => void;
}

/** Navigation tab row; one tab per task. */
export function TabBar({ tasks, activeKey, setActiveKey }: TabBarProps) {
  return (
    <nav
      data-tour="tabs"
      style={{
        position: "relative",
        zIndex: 3,
        maxWidth: 1280,
        margin: "24px auto 0",
        padding: "0 40px",
        borderBottom: "1px solid var(--vf-panel-stroke)",
      }}
    >
      <div
        style={{
          display: "flex",
          gap: 4,
          marginBottom: -1,
          overflowX: "auto",
          scrollbarWidth: "none",
        }}
      >
        {tasks.map((t) => {
          const active = t.key === activeKey;
          return (
            <button
              key={t.key}
              type="button"
              onClick={() => setActiveKey(t.key)}
              style={{
                padding: "14px 22px",
                whiteSpace: "nowrap",
                flexShrink: 0,
                background: active
                  ? `linear-gradient(180deg, ${t.accent}11 0%, transparent 100%)`
                  : "transparent",
                border: "none",
                borderBottom: `2px solid ${active ? t.accent : "transparent"}`,
                color: active ? "var(--vf-text)" : "var(--vf-text-dim)",
                fontFamily: "var(--font-mono)",
                fontSize: 12,
                letterSpacing: "0.14em",
                textTransform: "uppercase",
                position: "relative",
                transition: "all 220ms ease",
                cursor: "pointer",
              }}
            >
              <span
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  gap: 10,
                }}
              >
                <span
                  style={{
                    width: 6,
                    height: 6,
                    borderRadius: "50%",
                    background: t.accent,
                    boxShadow: active ? `0 0 10px ${t.accent}` : "none",
                    opacity: active ? 1 : 0.5,
                    flexShrink: 0,
                  }}
                />
                {t.label}
              </span>
              {active && (
                <span
                  style={{
                    position: "absolute",
                    bottom: -1,
                    left: "15%",
                    right: "15%",
                    height: 2,
                    background: t.accent,
                    boxShadow: `0 0 14px ${t.accent}`,
                    borderRadius: 999,
                  }}
                />
              )}
            </button>
          );
        })}
        <button
          type="button"
          style={{
            padding: "14px 18px",
            background: "transparent",
            border: "none",
            color: "var(--vf-text-muted)",
            fontFamily: "var(--font-mono)",
            fontSize: 18,
            letterSpacing: "0.2em",
          }}
        >
          ···
        </button>
      </div>
    </nav>
  );
}
