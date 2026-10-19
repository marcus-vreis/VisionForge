import { Monitor, Moon, Sun } from "lucide-react";
import { useEffect, useState } from "react";

import { useT } from "../i18n/useT";
import {
  THEME_PREFS,
  applyTheme,
  readThemePref,
  resolveTheme,
  storeThemePref,
  systemPrefersDark,
  watchSystemTheme,
  type ThemePref,
} from "../lib/theme";

const ICONS = { system: Monitor, light: Sun, dark: Moon } as const;

/** Automatic, light or dark, next to the language pair (ADR-125). */
export function ThemeSwitch() {
  const t = useT();
  const [pref, setPref] = useState<ThemePref>(readThemePref);
  const [systemDark, setSystemDark] = useState(systemPrefersDark);

  useEffect(() => watchSystemTheme(setSystemDark), []);
  useEffect(() => applyTheme(resolveTheme(pref, systemDark)), [pref, systemDark]);

  const choose = (next: ThemePref) => {
    setPref(next);
    storeThemePref(next);
  };

  return (
    // Three real buttons, like the language pair: aria-pressed says which is on.
    <div
      role="group"
      aria-label={t.theme.label}
      style={{
        display: "flex",
        alignItems: "center",
        gap: 2,
        padding: "0 4px",
        background: "rgb(var(--lt-ink, 255 255 255) / 0.025)",
        border: "1px solid var(--vf-panel-stroke)",
        borderRadius: 10,
        animation: "fadeUp 700ms ease both",
      }}
    >
      {THEME_PREFS.map((code) => {
        const Icon = ICONS[code];
        const on = code === pref;
        return (
          <button
            key={code}
            type="button"
            aria-label={t.theme.names[code]}
            aria-pressed={on}
            title={t.theme.names[code]}
            onClick={() => choose(code)}
            style={{
              display: "flex",
              padding: "8px 5px",
              background: "none",
              border: "none",
              color: on ? "var(--accent-vf)" : "var(--vf-text-muted)",
              cursor: "pointer",
            }}
          >
            <Icon size={14} strokeWidth={on ? 2.2 : 1.6} aria-hidden />
          </button>
        );
      })}
    </div>
  );
}
