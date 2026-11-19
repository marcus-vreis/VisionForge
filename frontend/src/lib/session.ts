/** What this server offers to whoever is asking (ADR-216).
 *
 * `GET /api/session` says whether the server is shared (`--server`), who the
 * login proxy says is asking, where their datasets may live, and which routes
 * answer 404 there. The screen hides what is closed instead of offering a
 * button that fails.
 *
 * The session is read once, when the page loads, and kept here, outside React:
 * the components that hide something are spread over the whole tree, and a
 * provider would have to wrap every test that renders one of them. A local
 * install, an older server and a failed read all mean the same: nothing is
 * closed. The server is what refuses; this only keeps the screen honest.
 */
import { useSyncExternalStore } from "react";

export interface Session {
  server_mode: boolean;
  /** The account the login proxy named; null on a local install. */
  user: string | null;
  /** The profile folder of that account; null on a local install. */
  profile: string | null;
  /** Where this person's datasets may live: their own folder, then the shared ones. */
  dataset_dirs: string[];
  /** The routes that answer 404 here, as "METHOD /api/template". */
  closed: string[];
  /** Classification sends its strategies inside the run, as `block`: the
   *  values this server refuses. */
  closed_blocks: string[];
}

export const LOCAL_SESSION: Session = {
  server_mode: false,
  user: null,
  profile: null,
  dataset_dirs: [],
  closed: [],
  closed_blocks: [],
};

/** The routes the screen asks about, by what they do. A test on the server
 *  side checks that every one of them is a route the server knows. */
export const ROUTE = {
  pickFolder: "POST /api/dataset/pick",
  pickYaml: "POST /api/detection/dataset/pick_yaml",
  pickCheckpoint: "POST /api/checkpoint/pick",
  sampleDataset: "POST /api/sample-dataset",
  datasetDownload: "POST /api/dataset/download",
  datasetLab: "POST /api/explorer/summary",
  datasetAudit: "POST /api/dataset/audit",
  previewAugment: "POST /api/dataset/preview_augment",
  previewPreprocess: "POST /api/dataset/preview_preprocess",
  colab: "GET /api/colab/status",
  customTasks: "GET /api/custom/{key}/schema",
  customTaskManage: "POST /api/custom/{key}/hide",
  credentials: "GET /api/credentials",
  profiles: "GET /api/profiles",
  profileCreate: "POST /api/profiles",
  profileUpdate: "PUT /api/profiles/{slug}",
  updateApply: "POST /api/update/apply",
  runResume: "POST /api/runs/{run_id}/resume",
  runReveal: "POST /api/runs/{run_id}/reveal",
  runTest: "POST /api/runs/{run_id}/test",
  runGradcam: "POST /api/runs/{run_id}/gradcam",
  runBatchPredict: "POST /api/runs/{run_id}/batch_predict",
  runExportOnnx: "POST /api/runs/{run_id}/export_onnx",
  runBundle: "GET /api/runs/{run_id}/bundle",
  runMarkdown: "GET /api/runs/{run_id}/export_md",
  hubExport: "GET /api/hub/preview",
  githubExport: "GET /api/github/preview",
  zenodoExport: "GET /api/zenodo/preview",
  dvcExport: "GET /api/dvc/preview",
  mlflowRetry: "POST /api/runs/{run_id}/mlflow/retry",
  wandbRetry: "POST /api/wandb/runs/{run_id}/retry",
  cometRetry: "POST /api/comet/runs/{run_id}/retry",
} as const;

/** The route a strategy of a task submits to, for the strategies that have one
 *  of their own. Classification sends cross-validation, the search and the
 *  comparison through its `run`, inside the config, so those have none. */
const STRATEGY_PATH: Record<string, string> = {
  cv: "cv",
  sweep: "sweep",
  compare: "compare",
  replicates: "replicates",
  replicatedComparison: "replicated-comparison",
};

export function strategyRoute(task: string, strategy: string): string | null {
  const path = STRATEGY_PATH[strategy];
  return path ? `POST /api/${task}/${path}` : null;
}

function strings(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string")
    : [];
}

/** The session as the server sent it; anything else reads as a local install. */
export function parseSession(raw: unknown): Session {
  if (typeof raw !== "object" || raw === null) return LOCAL_SESSION;
  const body = raw as Record<string, unknown>;
  if (body.server_mode !== true) return LOCAL_SESSION;
  return {
    server_mode: true,
    user: typeof body.user === "string" ? body.user : null,
    profile: typeof body.profile === "string" ? body.profile : null,
    dataset_dirs: strings(body.dataset_dirs),
    closed: strings(body.closed),
    closed_blocks: strings(body.closed_blocks),
  };
}

/** True when `route` answers for this session. */
export function routeOpen(session: Session, route: string | null): boolean {
  return route === null || !session.closed.includes(route);
}

/** The strategies of `task` this session can submit; the first always stays. */
export function openStrategies<S extends { value: string }>(
  session: Session,
  task: string,
  strategies: readonly S[],
): S[] {
  return strategies.filter(
    (s) =>
      routeOpen(session, strategyRoute(task, s.value)) &&
      !(task === "classification" && session.closed_blocks.includes(s.value)),
  );
}

/** Whether the datasets sheet has anything in it: the download card or the
 *  Laboratory. With neither, its button and the tour's stop on it go away. */
export function datasetsOffered(session: Session): boolean {
  return (
    routeOpen(session, ROUTE.datasetDownload) || routeOpen(session, ROUTE.datasetLab)
  );
}

let current: Session = LOCAL_SESSION;
const listeners = new Set<() => void>();

export function getSession(): Session {
  return current;
}

export function setSession(next: Session): void {
  current = next;
  listeners.forEach((notify) => notify());
}

/** Read the session before the page is first drawn, without holding it for long.
 *
 * Drawn first, the page would show (and call) what a shared server has closed
 * until the answer arrived. A server that takes longer than `waitMs` gets the
 * page anyway, and the session is applied when it comes.
 */
export async function bootSession(
  read: () => Promise<Session>,
  waitMs = 3000,
): Promise<void> {
  const arrived = read().then(setSession);
  await Promise.race([
    arrived,
    new Promise<void>((resolve) => setTimeout(resolve, waitMs)),
  ]);
}

function subscribe(notify: () => void): () => void {
  listeners.add(notify);
  return () => listeners.delete(notify);
}

export function useSession(): Session {
  return useSyncExternalStore(subscribe, getSession, getSession);
}

export function useRouteOpen(route: string | null): boolean {
  return routeOpen(useSession(), route);
}

export function useOpenStrategies<S extends { value: string }>(
  task: string,
  strategies: readonly S[],
): S[] {
  return openStrategies(useSession(), task, strategies);
}
