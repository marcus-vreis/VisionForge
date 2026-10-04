import { createContext } from "react";
import type { Lang } from "./lang";
import type { Dict } from "./pt";

export interface LanguageContextValue {
  lang: Lang;
  /** The active dictionary. */
  t: Dict;
  /** BCP 47 tag for toLocaleDateString and friends. */
  locale: string;
  setLang: (lang: Lang) => void;
}

export const LanguageContext = createContext<LanguageContextValue | null>(null);
