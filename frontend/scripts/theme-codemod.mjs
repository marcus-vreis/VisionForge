// Rewrites dark-only color literals so the light theme can retarget them (ADR-125).
//
//   node scripts/theme-codemod.mjs            rewrite src/ in place
//   node scripts/theme-codemod.mjs --check    exit 1 if a literal is left to convert
//
// The dark theme renders the same numbers: each rewrite keeps the original value
// as the fallback of a `--lt-*` variable that only the light theme defines.
import { readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const SRC = new URL("../src/", import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1");

// A tone this light was picked to read on the dark background.
const MIN_LIGHTNESS = 0.74;
// Light theme: L' = L - (2L - 1.3), a mirror around 0.65 (0.85 -> 0.45).
const MIRROR = 1.3;
// Colors that end up in SVG attributes, where a CSS variable does not resolve
// (training-story/tone.ts says why): they stay literal in both themes.
const SVG_COLOR_FILES = new Set(["tone.ts", "task-accent.ts"]);
// Near-black channels are a surface (panel, well, scrim), not ink.
const MAX_SURFACE_CHANNEL = 22;

function num(text) {
  return Number.parseFloat(text);
}

function convertRgba(_m, r, g, b, alpha) {
  const [R, G, B] = [r, g, b].map(Number);
  if (R === 255 && G === 255 && B === 255) return `rgb(var(--lt-ink, 255 255 255) / ${alpha})`;
  if (R === 0 && G === 0 && B === 0) return `rgb(var(--lt-shade, 0 0 0) / ${alpha})`;
  if (Math.max(R, G, B) <= MAX_SURFACE_CHANNEL) {
    return `rgb(var(--lt-surface, ${R} ${G} ${B}) / ${alpha})`;
  }
  return _m;
}

function convertOklch(m, l, c, h) {
  const L = num(l);
  if (!(L >= MIN_LIGHTNESS)) return m;
  const delta = Number((2 * L - MIRROR).toFixed(3));
  return `oklch(calc(${l} - var(--lt-flip, 0) * ${delta}) ${c} ${h})`;
}

const RGBA = /rgba\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*([^)]+?)\s*\)/g;
// Solid tones only: a hue followed straight by ")" has no alpha.
const OKLCH = /oklch\(\s*(\d?\.\d+|1)\s+(\d?\.\d+|0)\s+(\d+(?:\.\d+)?)\s*\)/g;

export function convert(source) {
  return source.replace(RGBA, convertRgba).replace(OKLCH, convertOklch);
}

function* files(dir) {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) yield* files(path);
    else if (/\.(tsx?|css)$/.test(entry.name) && !/\.test\.tsx?$/.test(entry.name)) {
      // index.css carries the hand-written theme blocks.
      if (entry.name === "index.css" || SVG_COLOR_FILES.has(entry.name)) continue;
      yield path;
    }
  }
}

// Imported by the tests for `convert`; only a direct run touches files.
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const check = process.argv.includes("--check");
  let pending = 0;
  for (const path of files(SRC)) {
    const before = readFileSync(path, "utf8");
    const after = convert(before);
    if (after === before) continue;
    pending += 1;
    if (check) console.log(`dark-only color literal: ${path}`);
    else writeFileSync(path, after);
  }
  if (check && pending) process.exit(1);
  if (!check) console.log(`rewrote ${pending} files`);
}
