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

/** The header that tells the server which language to write its messages in (ADR-116). */
export const LANG_HEADER = "X-VF-Lang";

/** The header every API call carries.
 *
 * Unlike the profile header, it is always sent: the server's default is
 * Portuguese, so even the default interface language has to say so when it is
 * English, and saying it for Portuguese too costs nothing and keeps one rule.
 */
export function langHeaders(lang: Lang): Record<string, string> {
  return { [LANG_HEADER]: lang };
}

/** The BCP 47 tag for dates, numbers and <html lang>. */
export function localeOf(lang: Lang): string {
  return lang === "pt" ? "pt-BR" : "en-US";
}
