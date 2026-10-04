/** The interface languages. Portuguese is the source the English is kept in step with. */
export type Lang = "pt" | "en";

const KEY = "vf.lang";

/**
 * The language to open in: the one chosen before, else the browser's.
 *
 * Any Portuguese locale gets Portuguese; everything else gets English, which
 * is the language a researcher outside Brazil is most likely to read.
 */
export function initialLang(stored: string | null, browser: string | undefined): Lang {
  if (stored === "pt" || stored === "en") return stored;
  return browser?.toLowerCase().startsWith("pt") ? "pt" : "en";
}

export function readStoredLang(): string | null {
  try {
    return localStorage.getItem(KEY);
  } catch {
    return null;
  }
}

export function storeLang(lang: Lang): void {
  try {
    localStorage.setItem(KEY, lang);
  } catch {
    /* storage unavailable — the choice lasts until the tab closes */
  }
}

/** The BCP 47 tag for dates, numbers and <html lang>. */
export function localeOf(lang: Lang): string {
  return lang === "pt" ? "pt-BR" : "en-US";
}
