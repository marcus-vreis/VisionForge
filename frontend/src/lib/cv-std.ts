/** Which divisor a K-fold run's standard deviation used.
 *
 * Runs written before ADR-111 divided by n (`ddof` 0); the sample std (n−1,
 * `ddof` 1) is what newer ones record in `std_ddof` beside the aggregate. A std
 * is only comparable between two runs that agree on it, so the detail view says
 * which one it shows. A run that records nothing used the old one.
 */
export function stdDdof(aggregate: { std_ddof?: unknown } | null | undefined): 0 | 1 {
  return aggregate?.std_ddof === 1 ? 1 : 0;
}
