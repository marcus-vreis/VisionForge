import { useId } from "react";

export interface FieldFeedbackProps {
  hint?: string;
  help?: string;
  error?: string;
}

/** Give each field stable label and feedback targets independent of its text. */
export function useFieldAccessibility({ hint, help, error }: FieldFeedbackProps) {
  const id = useId();
  const hintId = hint ? `${id}-hint` : undefined;
  const helpId = help ? `${id}-help` : undefined;
  const errorId = error ? `${id}-error` : undefined;
  return {
    id,
    labelId: `${id}-label`,
    hintId,
    helpId,
    errorId,
    "aria-describedby": [hintId, helpId, errorId].filter(Boolean).join(" ") || undefined,
    "aria-invalid": error ? true : undefined,
  } as const;
}
