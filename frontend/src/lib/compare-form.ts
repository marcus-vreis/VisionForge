/** Body of `POST /{task}/compare` beside the base config (ADR-044). */
export interface ComparePayload {
  model_names: string[];
  metric: string;
}

/** Add or remove `name`, keeping the order the options are listed in. */
export function toggleModel(
  selected: string[],
  name: string,
  order: string[],
): string[] {
  const next = selected.includes(name)
    ? selected.filter((n) => n !== name)
    : [...selected, name];
  return order.filter((n) => next.includes(n));
}

/** The comparison request body; the server refuses fewer than two models. */
export function buildComparePayload(
  modelNames: string[],
  metric: string,
): ComparePayload {
  return { model_names: [...modelNames], metric };
}
