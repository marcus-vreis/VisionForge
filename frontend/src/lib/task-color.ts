/**
 * The color of a researcher's own task in the interface (ADR-137).
 *
 * A task registered with `@register_task` declares an accent (`#rrggbb`); the
 * stage takes its tokens from it, and the researcher can pick another one from
 * the header. The choice is per browser, like the theme and the language.
 */
import type { CSSProperties } from "react";

const KEY_PREFIX = "vf.taskColor.";
const HEX = /^#[0-9a-f]{6}$/i;

/** Swatches offered next to the free picker; spread around the hue circle. */
export const TASK_COLOR_PRESETS: readonly string[] = [
  "#2dd4bf",
  "#22c55e",
  "#84cc16",
  "#eab308",
  "#f97316",
  "#ef4444",
  "#ec4899",
  "#a855f7",
  "#6366f1",
  "#0ea5e9",
];

/** The format the task registry declares and the color input returns. */
export function isHexColor(value: string): boolean {
  return HEX.test(value);
}

export function readTaskColor(key: string): string | null {
  try {
    const raw = localStorage.getItem(KEY_PREFIX + key);
    return raw && isHexColor(raw) ? raw : null;
  } catch {
    return null;
  }
}

/** Saves the chosen color, or forgets it (back to the registered one) with null. */
export function saveTaskColor(key: string, color: string | null): void {
  try {
    if (color && isHexColor(color)) localStorage.setItem(KEY_PREFIX + key, color);
    else localStorage.removeItem(KEY_PREFIX + key);
  } catch {
    /* storage unavailable — the color lasts until the tab closes */
  }
}

/** The task list with each chosen color in place of the registered accent. */
export function withTaskColors<T extends { key: string; accent: string }>(
  tasks: readonly T[],
  chosen: Readonly<Record<string, string>>,
  isResearcherTask: (task: T) => boolean,
): T[] {
  return tasks.map((task) =>
    isResearcherTask(task) && chosen[task.key] ? { ...task, accent: chosen[task.key] } : task,
  );
}

/** The stage's accent tokens (index.css) from a single color, for a task index.css
 *  has no block for. color-mix keeps the derived tones in the color's own hue. */
export function stageAccentVars(color: string): CSSProperties {
  return {
    "--accent-vf": color,
    "--accent-2": `color-mix(in oklch, ${color} 82%, black)`,
    "--accent-soft": `color-mix(in oklch, ${color} 14%, transparent)`,
    "--accent-glow": `color-mix(in oklch, ${color} 40%, transparent)`,
  } as CSSProperties;
}
