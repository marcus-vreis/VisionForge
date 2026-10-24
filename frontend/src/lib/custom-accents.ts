import { createContext, useContext } from "react";

/** The colors of the researcher's tasks by run task name (`custom:<key>`), for the
 *  history views, which sit far below the tab bar that owns the choice (ADR-137). */
export const CustomAccentsContext = createContext<Readonly<Record<string, string>>>({});

export function useCustomAccents(): Readonly<Record<string, string>> {
  return useContext(CustomAccentsContext);
}
