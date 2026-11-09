// What the DVC panel says, as pure functions of the server's answers (ADR-176).
import type { DvcExportResult, DvcRepositoryState } from "../api/client";
import type { Dict } from "../i18n/pt";

/** What the page says about the folder before anything is written. */
export function repositoryNotes(repository: DvcRepositoryState, d: Dict["dvc"]): string[] {
  const notes: string[] = [];
  if (repository.state !== "ready") notes.push(d.state[repository.state]);
  if (!repository.tools.git) notes.push(d.noGit);
  if (!repository.tools.dvc) notes.push(d.noDvc);
  if (canPrepare(repository)) notes.push(d.prepareHint);
  if (repository.state === "ready") notes.push(repository.remote ? d.remote(repository.remote) : d.noRemote);
  return notes;
}

/** Whether "Prepare folder" can make this path a git repository with DVC. */
export function canPrepare(repository: DvcRepositoryState): boolean {
  return repository.state !== "ready" && repository.state !== "not_dir"
    && repository.tools.git && repository.tools.dvc;
}

/** The commit, where the files are, and what happened to the weights. */
export function resultLines(result: DvcExportResult, d: Dict["dvc"]): string[] {
  const lines = [result.commit ? d.committed(result.commit.slice(0, 7)) : d.unchanged, d.savedAt(result.path)];
  if (result.pushed === true) lines.push(d.pushed);
  if (result.pushed === false) lines.push(d.pushFailed);
  return lines;
}
