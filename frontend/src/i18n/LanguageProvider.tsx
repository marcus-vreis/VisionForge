import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import { LanguageContext } from "./context";
import { en } from "./en";
import { initialLang, localeOf, readStoredLang, storeLang, type Lang } from "./lang";
import { pt, type Dict } from "./pt";
import { pageTitle, useSession } from "../lib/session";

const DICTS: Record<Lang, Dict> = { pt, en };

export function LanguageProvider({ children }: { children: ReactNode }) {
  const [lang, setLangState] = useState<Lang>(() =>
    initialLang(readStoredLang(), typeof navigator === "undefined" ? undefined : navigator.language),
  );

  // Screen readers and the browser's own translate prompt read <html lang>; the
  // tab title is worded in the same language (index.html's is English only),
  // and says "local" only where the page is not a shared server.
  const session = useSession();
  useEffect(() => {
    document.documentElement.lang = localeOf(lang);
    document.title = pageTitle(DICTS[lang].app, session);
  }, [lang, session]);

  const setLang = useCallback((next: Lang) => {
    setLangState(next);
    storeLang(next);
  }, []);

  const value = useMemo(
    () => ({ lang, t: DICTS[lang], locale: localeOf(lang), setLang }),
    [lang, setLang],
  );

  return <LanguageContext.Provider value={value}>{children}</LanguageContext.Provider>;
}
