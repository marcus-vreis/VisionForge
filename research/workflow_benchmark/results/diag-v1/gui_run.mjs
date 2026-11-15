// Diagnostic driver: same path as gui-v1/gui_run.mjs, then one of three client states.
// usage: node gui_run.mjs <server|hidden|visible> <config.yaml> <tab> <port> <timing.json>
import { createRequire } from "node:module";
import { writeFileSync, mkdtempSync } from "node:fs";
import { spawn, execSync } from "node:child_process";
import { tmpdir } from "node:os";
import { join } from "node:path";
const require = createRequire("~/Documents/VisionForge/frontend/");
const { chromium } = require("@playwright/test");
const [mode, config, tab, port, timing] = process.argv.slice(2);
const THROTTLE = ["--disable-backgrounding-occluded-windows", "--disable-renderer-backgrounding", "--disable-background-timer-throttling"];
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
// Playwright keeps a page "visible" while attached, so hidden runs in a plain Chrome it only visits:
// it minimizes the window and disconnects, and a raw CDP probe then reads the page state.
const DEBUG = 9333;
let chrome, browser, page;
if (mode === "hidden") {
  chrome = spawn("C:/Program Files/Google/Chrome/Application/chrome.exe", [`--remote-debugging-port=${DEBUG}`, `--user-data-dir=${mkdtempSync(join(tmpdir(), "diag-"))}`, "--no-first-run", "--window-size=1366,768", "about:blank"], { stdio: "ignore" });
  await sleep(4000);
  browser = await chromium.connectOverCDP(`http://127.0.0.1:${DEBUG}`);
  page = browser.contexts()[0].pages()[0];
} else {
  browser = await chromium.launch({ headless: false, channel: "chrome", args: THROTTLE });
  page = await browser.newPage({ viewport: { width: 1366, height: 768 } });
}
const probe = async () => {
  const t = (await (await fetch(`http://127.0.0.1:${DEBUG}/json`)).json()).find((x) => x.type === "page");
  const ws = new WebSocket(t.webSocketDebuggerUrl);
  await new Promise((r) => { ws.onopen = r; });
  const out = new Promise((r) => { ws.onmessage = (m) => { const d = JSON.parse(m.data); if (d.id === 1) r(d.result.result.value); }; });
  ws.send(JSON.stringify({ id: 1, method: "Runtime.evaluate", params: { awaitPromise: true, returnByValue: true, expression: "new Promise(res=>{let n=0;const t=()=>{n++;requestAnimationFrame(t)};requestAnimationFrame(t);setTimeout(()=>res({visibility:document.visibilityState,raf_2s:n}),2000)})" } }));
  const v = await out; ws.close(); return v;
};
const closeGuide = page.getByRole("button", { name: "Fechar o guia" });
await page.addLocatorHandler(closeGuide, () => closeGuide.click());
await page.goto(`http://127.0.0.1:${port}/`);
await page.getByText("Qual é o seu nome?").waitFor({ timeout: 60000 });
const field = page.getByRole("textbox", { name: "Seu nome" });
for (let i = 0; i < 40; i++) {
  try { await field.fill("Bench", { timeout: 1000 }); if ((await field.inputValue()) === "Bench") break; } catch {}
  await page.waitForTimeout(500);
}
await page.getByRole("button", { name: /Entrar/ }).click();
await field.waitFor({ state: "hidden" });
await page.locator("[data-tour='tabs']").getByText(tab, { exact: true }).click();
await page.locator("input[type=file]").first().setInputFiles(config);
await page.waitForTimeout(1500);
await page.getByRole("navigation", { name: "Etapas da configuração" }).getByRole("button", { name: "Revisão" }).click();
const t0 = Date.now();
await page.locator("[data-tour='train']").click();
const confirm = page.getByRole("button", { name: /Treinar mesmo assim|Continuar|Confirmar/ });
await page.waitForTimeout(1500);
if (await confirm.count()) await confirm.first().click();
const queue = async () => (await fetch(`http://127.0.0.1:${port}/api/queue`)).json();
for (let i = 0; i < 120 && !(await queue()).active; i++) await sleep(500);
const info = { mode };
if (mode === "server") await browser.close();
if (mode === "hidden") {
  const cdp = await page.context().newCDPSession(page);
  const { windowId } = await cdp.send("Browser.getWindowForTarget");
  await cdp.send("Browser.setWindowBounds", { windowId, bounds: { windowState: "minimized" } });
  await browser.close();
  await sleep(5000);
  Object.assign(info, await probe());
}
for (;;) {
  const q = await queue();
  if (!q.active && !q.pending.length) break;
  await sleep(2000);
}
writeFileSync(timing, JSON.stringify({ ...info, click_to_done_s: (Date.now() - t0) / 1000 }, null, 1));
if (mode === "visible") await browser.close();
if (mode === "hidden") execSync(`taskkill /PID ${chrome.pid} /T /F`, { stdio: "ignore" });
process.exit(0);
