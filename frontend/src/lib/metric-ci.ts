import type { MetricCI } from "../types/run";

/** Look up the bootstrap interval for a metrics-block key (ADR-074/076).
 *
 * Every entry in `metric_cis` is measured on the **test** split, so only a
 * `test_`-prefixed metric can have one. That restriction is load-bearing rather
 * than tidiness: segmentation's run.json carries both `miou` (the best
 * *validation* score from training) and `test_miou`, and a lookup that also
 * accepted the bare name pinned the test interval onto the validation number —
 * an interval bracketing a value it was not computed from.
 */
export function metricCi(
  cis: Record<string, MetricCI> | undefined,
  metricsKey: string,
): MetricCI | undefined {
  if (!cis || !metricsKey.startsWith("test_")) return undefined;
  return cis[metricsKey.slice("test_".length)];
}

/** `0.7506 [0.7294, 0.7713]` — the citable form for one metric. */
export function formatWithCi(value: number, ci: MetricCI | undefined, digits = 4): string {
  const point = value.toFixed(digits);
  if (!ci) return point;
  return `${point} [${ci.ci_low.toFixed(digits)}, ${ci.ci_high.toFixed(digits)}]`;
}
