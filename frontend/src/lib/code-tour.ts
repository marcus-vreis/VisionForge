import { ApiError, fetchCodeTourTasks, type CodeSection } from "../api/client";

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

/** The words the library notice is made of (the `codeTour` dictionary). */
export interface LibraryWords {
  library: (name: string, version: string, license: string) => string;
  notesResolved: (resolved: number, total: number) => string;
}

/**
 * The line above a section whose code is a library's (ADR-194): which library,
 * the installed version, its license, and how many notes still found their
 * line in that version. Null for VisionForge's own code.
 */
export function libraryNotice(section: CodeSection, words: LibraryWords): string | null {
  if (!section.library) return null;
  const { name, version, license } = section.library;
  return `${words.library(name, version, license)} ${words.notesResolved(section.notes_resolved, section.notes_total)}`;
}
