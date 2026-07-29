import type { MetricCI } from "../types/run";

/** Look up the bootstrap interval for a metrics-block key (ADR-074).
 *
 * The two blocks are keyed differently on purpose: `metrics` prefixes the
 * test-split entries (`test_accuracy`) because it also holds training numbers
 * like `best_val_loss`, while `metric_cis` only ever describes the test split
 * and keys by the bare metric name. This bridges the two so a caller can ask
 * with whichever key it is already iterating.
 */
export function metricCi(
  cis: Record<string, MetricCI> | undefined,
  metricsKey: string,
): MetricCI | undefined {
  if (!cis) return undefined;
  const bare = metricsKey.startsWith("test_") ? metricsKey.slice("test_".length) : metricsKey;
  return cis[bare];
}

/** `0.7506 [0.7294, 0.7713]` — the citable form for one metric. */
export function formatWithCi(value: number, ci: MetricCI | undefined, digits = 4): string {
  const point = value.toFixed(digits);
  if (!ci) return point;
  return `${point} [${ci.ci_low.toFixed(digits)}, ${ci.ci_high.toFixed(digits)}]`;
}
