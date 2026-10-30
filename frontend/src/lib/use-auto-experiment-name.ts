import { useEffect } from "react";
import { isAutoName, rememberSuggestion } from "./experiment-name";

/** Keeps the experiment name on the suggestion while the researcher has not
 *  written their own (ADR-155). Runs when the suggestion changes, not on every
 *  keystroke, so clearing the field to type a new name is not undone. */
export function useAutoExperimentName(
  taskKey: string,
  name: string,
  suggestion: string,
  setName: (next: string) => void,
  defaults: readonly string[],
): void {
  useEffect(() => {
    if (isAutoName(taskKey, name, defaults) && name !== suggestion) {
      setName(suggestion);
    }
    rememberSuggestion(taskKey, suggestion);
    // `name` is read, not followed: see the docstring.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [taskKey, suggestion]);
}
