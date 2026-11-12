/** Whether a finished run's results belong on the open task tab.
 *
 * `runTaskKey` is the task the run was submitted from. Results are hidden only
 * on the tab of another task: a run whose task is unknown, or has no tab (a
 * queue entry of a task removed since), stays visible wherever the user is.
 */
export function resultsOnTab(
  runTaskKey: string | null,
  activeKey: string,
  tabKeys: readonly string[],
): boolean {
  if (runTaskKey === null || !tabKeys.includes(runTaskKey)) return true;
  return runTaskKey === activeKey;
}
