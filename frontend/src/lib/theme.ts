/**
 * The light and dark themes (ADR-125).
 *
 * The choice is per browser, like the language: `system` (the default) follows
 * the operating system, `light` and `dark` pin one. The resolved theme goes on
 * `<html data-theme>`; index.css keys the light palette off it. index.html runs
 * the same resolution inline before the first paint, so keep the key in step.
 */
export type ThemePref = "system" | "light" | "dark";
export type Theme = "light" | "dark";

export const THEME_KEY = "vf.theme";
export const THEME_PREFS: readonly ThemePref[] = ["system", "light", "dark"];

const DARK_QUERY = "(prefers-color-scheme: dark)";

/** A stored value read back as a choice; anything unknown is `system`. */
export function parseThemePref(stored: string | null): ThemePref {
  return stored === "light" || stored === "dark" ? stored : "system";
}

/** The theme to paint: the pinned one, or the system's. */
export function resolveTheme(pref: ThemePref, systemDark: boolean): Theme {
  if (pref === "system") return systemDark ? "dark" : "light";
  return pref;
}

export function readThemePref(): ThemePref {
  try {
    return parseThemePref(localStorage.getItem(THEME_KEY));
  } catch {
    return "system";
  }
}

export function storeThemePref(pref: ThemePref): void {
  try {
    if (pref === "system") localStorage.removeItem(THEME_KEY);
    else localStorage.setItem(THEME_KEY, pref);
  } catch {
    /* storage unavailable — the choice lasts until the tab closes */
  }
}

export function systemPrefersDark(): boolean {
  // The palette was dark-only before ADR-125: a browser that cannot say keeps it.
  return typeof window.matchMedia === "function" ? window.matchMedia(DARK_QUERY).matches : true;
}

export function applyTheme(theme: Theme): void {
  document.documentElement.dataset.theme = theme;
}

/** Calls `onChange` when the operating system switches theme; returns the unsubscribe. */
export function watchSystemTheme(onChange: (dark: boolean) => void): () => void {
  if (typeof window.matchMedia !== "function") return () => {};
  const query = window.matchMedia(DARK_QUERY);
  const listener = (event: MediaQueryListEvent) => onChange(event.matches);
  query.addEventListener("change", listener);
  return () => query.removeEventListener("change", listener);
}
