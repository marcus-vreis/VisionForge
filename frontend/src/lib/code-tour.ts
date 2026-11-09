import { ApiError, fetchCodeTourTasks } from "../api/client";

let tourTasks: Promise<string[]> | null = null;

/**
 * The tasks with a tour in the "Código" tab (ADR-173), asked of the server once
 * per page: it decides which tasks have one, so a new tour needs no change here.
 * A failed request is asked again the next time.
 */
export function loadTourTasks(): Promise<string[]> {
  tourTasks ??= fetchCodeTourTasks()
    .then((list) => list.map((item) => item.task))
    .catch((error: unknown) => {
      tourTasks = null;
      throw error;
    });
  return tourTasks;
}

/** Whether `taskKey` has a tour, once the list of tasks is known. */
export function tourStatus(taskKey: string, tasks: string[] | null): "loading" | "missing" | "supported" {
  if (tasks === null) return "loading";
  return tasks.includes(taskKey) ? "supported" : "missing";
}

/**
 * The server's reason when it has no tour for this run (a 404 with a message,
 * such as a task whose training is not this project's code), or null for any
 * other failure.
 */
export function unavailableReason(error: unknown): string | null {
  return error instanceof ApiError && error.status === 404 && error.message ? error.message : null;
}
