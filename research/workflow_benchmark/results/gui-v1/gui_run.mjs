// Drives one VisionForge run through the real GUI: import YAML, Revisão, Treinar, wait for the end.
// usage: node gui_run.mjs <config.yaml> <Classificação|Segmentação> <port> <timing.json> <shot.png>
import { createRequire } from "node:module";
import { writeFileSync } from "node:fs";
const require = createRequire("~/Documents/VisionForge/frontend/");
const { chromium } = require("@playwright/test");
const [config, tab, port, timing, shot] = process.argv.slice(2);
const browser = await chromium.launch({
  headless: false,
  channel: "chrome",
  args: ["--disable-backgrounding-occluded-windows", "--disable-renderer-backgrounding", "--disable-background-timer-throttling"],
});
const page = await browser.newPage({ viewport: { width: 1366, height: 768 } });
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
const review = page.getByRole("navigation", { name: "Etapas da configuração" }).getByRole("button", { name: "Revisão" });
await review.click();
const train = page.locator("[data-tour='train']");
const t0 = Date.now();
await train.click();
const confirm = page.getByRole("button", { name: /Treinar mesmo assim|Continuar|Confirmar/ });
await page.waitForTimeout(1500);
if (await confirm.count()) await confirm.first().click();
const done = page.getByText("treino concluído", { exact: true });
const failed = page.getByText("treino falhou", { exact: true });
await Promise.race([done.waitFor({ timeout: 4 * 3600e3 }), failed.waitFor({ timeout: 4 * 3600e3 })]);
const t1 = Date.now();
const ok = await done.count() > 0;
await page.screenshot({ path: shot });
writeFileSync(timing, JSON.stringify({ ok, click_to_done_s: (t1 - t0) / 1000, t0, t1 }, null, 1));
await browser.close();
process.exit(ok ? 0 : 1);
