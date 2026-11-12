// Lists what renders while a run is active: CSS/WAAPI animations and rAF callers.
import { createRequire } from "node:module";
import { writeFileSync } from "node:fs";
const require = createRequire("~/Documents/VisionForge/frontend/");
const { chromium } = require("@playwright/test");
const [config, tab, port, outp] = process.argv.slice(2);
const THROTTLE = ["--disable-backgrounding-occluded-windows", "--disable-renderer-backgrounding", "--disable-background-timer-throttling"];
const browser = await chromium.launch({ headless: false, channel: "chrome", args: THROTTLE });
const page = await browser.newPage({ viewport: { width: 1366, height: 768 } });
await page.addInitScript(() => {
  const raf = window.requestAnimationFrame.bind(window); window.__raf = {};
  window.requestAnimationFrame = (cb) => { const k = (new Error().stack || "").split("\n").slice(2, 4).join(" < ").replace(/https?:\/\/[^/]+/g, ""); window.__raf[k] = (window.__raf[k] || 0) + 1; return raf(cb); };
});
const closeGuide = page.getByRole("button", { name: "Fechar o guia" });
await page.addLocatorHandler(closeGuide, () => closeGuide.click());
await page.goto(`http://127.0.0.1:${port}/`);
await page.getByText("Qual é o seu nome?").waitFor({ timeout: 60000 });
const field = page.getByRole("textbox", { name: "Seu nome" });
for (let i = 0; i < 40; i++) { try { await field.fill("Bench", { timeout: 1000 }); if ((await field.inputValue()) === "Bench") break; } catch {} await page.waitForTimeout(500); }
await page.getByRole("button", { name: /Entrar/ }).click();
await field.waitFor({ state: "hidden" });
await page.locator("[data-tour='tabs']").getByText(tab, { exact: true }).click();
await page.locator("input[type=file]").first().setInputFiles(config);
await page.waitForTimeout(1500);
await page.getByRole("navigation", { name: "Etapas da configuração" }).getByRole("button", { name: "Revisão" }).click();
await page.locator("[data-tour='train']").click();
const confirm = page.getByRole("button", { name: /Treinar mesmo assim|Continuar|Confirmar/ });
await page.waitForTimeout(1500);
if (await confirm.count()) await confirm.first().click();
await page.evaluate((names) => { const kill = () => document.getAnimations().forEach((a) => { if (names.includes(a.animationName)) a.cancel(); }); kill(); setInterval(kill, 1000); }, (process.env.KILL || "").split(","));
const queue = async () => (await fetch(`http://127.0.0.1:${port}/api/queue`)).json();
for (;;) { const q = await queue(); if (!q.active && !q.pending.length) break; await new Promise((r) => setTimeout(r, 2000)); }
await browser.close(); process.exit(0);
const r = await page.evaluate(() => new Promise((res) => {
  window.__raf = {};
  let n = 0; const t = () => { n++; requestAnimationFrame(t); }; requestAnimationFrame(t);
  setTimeout(() => {
    const anims = document.getAnimations().filter((a) => a.playState === "running").map((a) => {
      const e = a.effect?.target; const cs = e ? getComputedStyle(e) : null;
      return { name: a.animationName || a.transitionProperty || "waapi", tag: e?.tagName, cls: String(e?.className?.baseVal ?? e?.className ?? "").slice(0, 80), text: (e?.textContent || "").slice(0, 40), props: [...new Set(a.effect.getKeyframes().flatMap((k) => Object.keys(k)))].filter((k) => !["offset", "easing", "composite", "computedOffset"].includes(k)), dur: a.effect.getComputedTiming().duration };
    });
    const blur = [...document.querySelectorAll("*")].filter((e) => { const c = getComputedStyle(e); return c.backdropFilter !== "none" || c.filter !== "none"; }).map((e) => ({ tag: e.tagName, cls: String(e.className?.baseVal ?? e.className ?? "").slice(0, 60), b: getComputedStyle(e).backdropFilter, f: getComputedStyle(e).filter }));
    res({ frames_5s: n, raf: window.__raf, anims, blur: blur.slice(0, 30), canvases: document.querySelectorAll("canvas").length });
  }, 5000);
}));
await page.screenshot({ path: outp + ".png" });
writeFileSync(outp, JSON.stringify(r, null, 1));
await fetch(`http://127.0.0.1:${port}/api/queue`).then((x) => x.json()).then((q) => q.active && fetch(`http://127.0.0.1:${port}/api/queue/${q.active.run_id}`, { method: "DELETE" }));
await browser.close();
