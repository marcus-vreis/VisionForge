/** Picking a Colab dataset folder inside this computer's "My Drive" (ADR-164).
 *
 * The server lists one folder at a time by its path under the "My Drive" root,
 * `/`-separated (`rel`); the panel walks it with a breadcrumb.
 */
import type { ColabDatasetSource } from "../api/client";

export interface DriveCrumb {
  name: string;
  rel: string;
}

/** The folders from the root down to `rel`, each with the `rel` that lists it. */
export function driveCrumbs(rel: string): DriveCrumb[] {
  const crumbs: DriveCrumb[] = [];
  for (const name of rel.split("/").filter(Boolean)) {
    const parent = crumbs.at(-1)?.rel;
    crumbs.push({ name, rel: parent ? `${parent}/${name}` : name });
  }
  return crumbs;
}

/** The `rel` of the sub-folder `name` of `rel`. */
export function childRel(rel: string, name: string): string {
  return rel ? `${rel}/${name}` : name;
}

/** The Drive source of a Colab run: the typed "My Drive" root and the picked
 *  folder go along only when given, so an empty form keeps the old request
 *  (the server then uses the form's own dataset folder). */
export function driveLaunchSource(myDriveRoot: string, picked: string): ColabDatasetSource {
  const root = myDriveRoot.trim();
  const path = picked.trim();
  return {
    kind: "drive",
    ...(root ? { my_drive_root: root } : {}),
    ...(path ? { path } : {}),
  };
}
