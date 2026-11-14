import { useEffect, useMemo, useState } from "react";
import { fetchCodeTour, type CodeSection, type CodeSettings } from "../../api/client";
import { loadTourTasks, tourStatus, unavailableReason } from "../../lib/code-tour";

/** Where the run's code stands: loading, there, or why it is not. */
export type TourLoad =
  | { kind: "loading" }
  | { kind: "failed" }
  /** The server lists no tour for this task (a researcher's own task). */
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
 */
export function useTour(taskKey: string, settings: CodeSettings | null): TourLoad {
  const [tasks, setTasks] = useState<string[] | null>(null);
  const [listFailed, setListFailed] = useState(false);
  const [loaded, setLoaded] = useState<Loaded | null>(null);
  const query = useMemo(() => JSON.stringify(settings ?? {}), [settings]);
  const status = tourStatus(taskKey, tasks);
  const request = `${taskKey}?${query}`;

  useEffect(() => {
    let alive = true;
    loadTourTasks()
      .then((list) => { if (alive) setTasks(list); })
      .catch(() => { if (alive) setListFailed(true); });
    return () => { alive = false; };
  }, []);

  useEffect(() => {
    if (status !== "supported") return;
    let alive = true;
    const key = `${taskKey}?${query}`;
    fetchCodeTour(taskKey, JSON.parse(query) as CodeSettings)
      .then((sections) => { if (alive) setLoaded({ request: key, load: { kind: "ready", sections } }); })
      .catch((error: unknown) => {
        if (!alive) return;
        const reason = unavailableReason(error);
        setLoaded({ request: key, load: reason ? { kind: "unavailable", reason } : { kind: "failed" } });
      });
    return () => { alive = false; };
  }, [status, taskKey, query]);

  if (listFailed) return { kind: "failed" };
  if (status === "missing") return { kind: "missing" };
  return loaded?.request === request ? loaded.load : { kind: "loading" };
}
