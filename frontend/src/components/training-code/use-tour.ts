import { useEffect, useMemo, useState } from "react";
import { fetchCodeTour, fetchCustomCodeTour, type CodeSection, type CodeSettings } from "../../api/client";
import { loadTourTasks, tourSupport, unavailableReason } from "../../lib/code-tour";

/** Where the run's code stands: loading, there, or why it is not. */
export type TourLoad =
  | { kind: "loading" }
  | { kind: "failed" }
  /** The server lists no tour for this task. */
  | { kind: "missing" }
  /** The server's own reason it has no tour for this run (a 404 with a message). */
  | { kind: "unavailable"; reason: string }
  | { kind: "ready"; sections: CodeSection[] };

interface Loaded {
  request: string;
  load: TourLoad;
}

/** The tour of `taskKey` for this run's settings (ADR-173, ADR-181).
 *
 * The server says which tasks have one and reads the code from the installed
 * package, so it is the code that trains. A failed list is asked again the next
 * time the tab opens (`loadTourTasks`).
 *
 * A researcher's own task (`custom`) is not in that list: its tour is read from
 * its own file by `GET /api/custom/{key}/code-tour`, whatever the settings.
 */
export function useTour(taskKey: string, settings: CodeSettings | null, custom = false): TourLoad {
  const [tasks, setTasks] = useState<string[] | null>(null);
  const [listFailed, setListFailed] = useState(false);
  const [loaded, setLoaded] = useState<Loaded | null>(null);
  const query = useMemo(() => JSON.stringify(settings ?? {}), [settings]);
  const status = tourSupport(taskKey, tasks, custom);
  const request = `${custom ? "custom:" : ""}${taskKey}?${query}`;

  useEffect(() => {
    if (custom) return;
    let alive = true;
    loadTourTasks()
      .then((list) => { if (alive) setTasks(list); })
      .catch(() => { if (alive) setListFailed(true); });
    return () => { alive = false; };
  }, [custom]);

  useEffect(() => {
    if (status !== "supported") return;
    let alive = true;
    const fetched = custom
      ? fetchCustomCodeTour(taskKey)
      : fetchCodeTour(taskKey, JSON.parse(query) as CodeSettings);
    fetched
      .then((sections) => { if (alive) setLoaded({ request, load: { kind: "ready", sections } }); })
      .catch((error: unknown) => {
        if (!alive) return;
        const reason = unavailableReason(error);
        setLoaded({ request, load: reason ? { kind: "unavailable", reason } : { kind: "failed" } });
      });
    return () => { alive = false; };
  }, [status, taskKey, query, custom, request]);

  if (!custom && listFailed) return { kind: "failed" };
  if (status === "missing") return { kind: "missing" };
  return loaded?.request === request ? loaded.load : { kind: "loading" };
}
