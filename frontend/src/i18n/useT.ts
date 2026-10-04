import { useContext } from "react";
import { LanguageContext, type LanguageContextValue } from "./context";
import type { Dict } from "./pt";

export function useI18n(): LanguageContextValue {
  const ctx = useContext(LanguageContext);
  if (!ctx) throw new Error("useI18n must be used inside <LanguageProvider>.");
  return ctx;
}

/** The active dictionary: `const t = useT(); t.common.save`. */
export function useT(): Dict {
  return useI18n().t;
}
