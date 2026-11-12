// Captures the README images from a running `visionforge gui` (issue #221).
//
// Driven by scripts/capture_docs_images.py, which starts the server in a
// throwaway work folder and home, and writes docs/images/README.md afterwards.
// Uses the Chrome already installed on the machine (channel "chrome"), as the
// local e2e suite does; set VF_DOCS_CHANNEL="" to use Playwright's chromium.
//
//   node scripts/capture-docs-images.mjs --base-url http://127.0.0.1:PORT --out DIR
import { chromium, expect } from "@playwright/test";
import { mkdirSync } from "node:fs";
import path from "node:path";
import { parseArgs } from "node:util";

const { values } = parseArgs({
  options: {
    "base-url": { type: "string" },
    out: { type: "string" },
  },
});
const baseURL = values["base-url"];
const outDir = values.out;
if (!baseURL || !outDir) {
  console.error("usage: capture-docs-images.mjs --base-url URL --out DIR");
  process.exit(2);
}
mkdirSync(outDir, { recursive: true });

// The size of the images the README has shown since v0.20.
const VIEWPORT = { width: 1440, height: 960 };
const VERSION_MARK = "data-docs-hidden-version";
const VERSION_TEXT = /·\s*v\d+\.\d+/;
const channel = process.env.VF_DOCS_CHANNEL ?? "chrome";

/**
 * Hides the " · vX.Y.Z" after "local ai studio" in the header, so an image
 * never goes stale on the next release, and asserts it is hidden in the DOM.
 *
 * React renders that suffix as a text node of its own; it is moved into a
 * span with display:none, so the rest of the subtitle stays as it is.
 */
async function hideVersion(page) {
  const pattern = VERSION_TEXT.source;
  // The header prints the version once the server's info arrives.
  await expect
    .poll(() =>
      page.evaluate(
        ([mark, source]) => {
          const re = new RegExp(source);
          return (
            document.querySelector(`[${mark}]`) !== null ||
            [...document.querySelectorAll("header *")].some((el) =>
              [...el.childNodes].some((n) => n.nodeType === Node.TEXT_NODE && re.test(n.nodeValue ?? "")),
            )
          );
        },
        [VERSION_MARK, pattern],
      ),
    )
    .toBe(true);
  await page.evaluate(
    ([mark, source]) => {
      const re = new RegExp(source);
      for (const el of document.querySelectorAll("header *")) {
        if (el.hasAttribute(mark)) continue;
        for (const node of [...el.childNodes]) {
          if (node.nodeType === Node.TEXT_NODE && re.test(node.nodeValue ?? "")) {
            const span = document.createElement("span");
            span.setAttribute(mark, "");
            span.style.display = "none";
            node.replaceWith(span);
            span.appendChild(node);
          }
        }
      }
    },
    [VERSION_MARK, pattern],
  );
  const hidden = page.locator(`[${VERSION_MARK}]`);
  await expect(hidden).toHaveCount(1);
  await expect(hidden).toBeHidden();
  // The subtitle that carried it no longer shows a version. (Other header
  // text may name one, e.g. the Colab CLI's pinned version: not ours.)
  const subtitle = await hidden.locator("xpath=..").innerText();
  if (/v\d+\.\d+/i.test(subtitle)) {
    throw new Error(`the header still shows the version: ${subtitle}`);
  }
}

async function shot(page, name) {
  await hideVersion(page);
  await page.mouse.move(0, 0);
  await page.evaluate(() => document.fonts.ready);
  // Lets entrance animations settle (the longest runs 700 ms).
  await page.waitForTimeout(1_200);
  await hideVersion(page);
  const file = path.join(outDir, name);
  await page.screenshot({ path: file });
  console.log(`captured ${file}`);
}

/** Gets past the first-visit welcome, as the e2e helper `enter` does. */
async function enter(page, name) {
  const closeGuide = page.getByRole("button", { name: "Fechar o guia" });
  await page.addLocatorHandler(closeGuide, () => closeGuide.click());
  await page.goto("/");
  await expect(page.getByText("Qual é o seu nome?")).toBeVisible();
  const nameField = page.getByRole("textbox", { name: "Seu nome" });
  await expect(async () => {
    await nameField.fill(name, { timeout: 1_000 });
    await expect(nameField).toHaveValue(name, { timeout: 500 });
  }).toPass();
  await page.getByRole("button", { name: /Entrar/ }).click();
  await expect(nameField).toBeHidden();
}

async function openTask(page, label) {
  await page.locator("[data-tour='tabs']").getByText(label, { exact: true }).click();
}

async function openStep(page, label) {
  const tab = page
    .getByRole("navigation", { name: "Etapas da configuração" })
    .getByRole("button", { name: label });
  await tab.click();
  await expect(tab).toHaveAttribute("aria-current", "step");
}

function field(page, label) {
  return page.getByLabel(label, { exact: true }).first();
}

async function useSampleDataset(page) {
  await openStep(page, "Dados");
  await expect(async () => {
    await page.getByRole("button", { name: "usar dataset de exemplo" }).click();
    await page.waitForFunction(
      () => [...document.querySelectorAll("input")].some((i) => i.value.includes("exemplo-classificacao")),
      null,
      { timeout: 5_000 },
    );
  }).toPass({ timeout: 30_000 });
  await expect(page.getByText(/Splits detectados/)).toBeVisible({ timeout: 30_000 });
}

async function classification(page) {
  await openTask(page, "Classificação");
  await useSampleDataset(page);
  await shot(page, "vf-classification.png");
  await openStep(page, "Estratégia");
  await expect(page.getByRole("button", { name: "Comparação replicada" })).toBeVisible();
  await shot(page, "vf-strategy.png");
}

/** The header's guide menu, open. */
async function guideMenu(page) {
  const button = page.getByTitle(/Abrir os guias/);
  await button.click();
  await expect(page.getByRole("menu")).toBeVisible();
  await shot(page, "vf-guide-menu.png");
  await page.keyboard.press("Escape");
  await expect(page.getByRole("menu")).toBeHidden();
}

/** The Colab corner as a fresh install sees it: no CLI and no Google account. */
async function colabCorner(page) {
  const chip = page.getByRole("button", { name: /^colab · / });
  const dialog = page.getByRole("dialog", { name: "Google Colab" });
  await chip.click();
  await expect(dialog).toBeVisible();
  await shot(page, "vf-colab.png");
  await chip.click();
  await expect(dialog).toBeHidden();
}

async function replicates(page) {
  await openTask(page, "Regressão");
  await openStep(page, "Estratégia");
  await page.getByRole("button", { name: "Réplicas", exact: true }).click();
  await expect(page.getByText(/Réplicas multi-seed/i).first()).toBeVisible();
  await shot(page, "vf-replicates.png");
}

/** Trains one CPU epoch on the sample dataset, then opens the "Código" tab. */
async function codeTab(page) {
  await openTask(page, "Classificação");
  await useSampleDataset(page);
  await openStep(page, "Treinamento");
  const epochs = field(page, "Épocas");
  await epochs.fill("1");
  await epochs.press("Enter");
  await expect(field(page, "Épocas")).toHaveValue("1");
  await openStep(page, "Modelo");
  // Random initialization: no weight download.
  const pretrained = page.getByRole("switch", { name: "Pesos pré-treinados", exact: true });
  if (await pretrained.isChecked()) await pretrained.click();
  await expect(pretrained).toHaveText(/desativado/i);
  const device = page.locator("[data-tour='device']");
  await device.getByRole("button").first().click();
  await device.getByRole("button", { name: /Treinar usando processador/ }).click();
  await openStep(page, "Revisão");
  await field(page, "Nome do experimento").fill("docs_code_tab");
  await shot(page, "vf-review.png");
  await page.locator("[data-tour='train']").click();
  const sheet = page.locator("[data-tour='training-sheet']");
  await expect(sheet).toBeVisible();
  await expect(sheet.getByText("treino concluído", { exact: true })).toBeVisible({ timeout: 300_000 });
  await sheet.getByRole("tab", { name: "Código" }).click();
  const panel = sheet.locator("[aria-labelledby='training-tab-code']");
  await expect(panel).toBeVisible();
  await expect(panel.getByText(/código real que treina/)).toBeVisible({ timeout: 30_000 });
  await shot(page, "vf-code-tab.png");
  await sheet.getByRole("tab", { name: "Log" }).click();
  await shot(page, "vf-training.png");
}

/** The results of the run `codeTab` trained, then its entry in the history. */
async function resultsAndHistory(page) {
  const sheet = page.locator("[data-tour='training-sheet']");
  await sheet.getByRole("button", { name: /Ver resultados/ }).click();
  await expect(sheet).toBeHidden();
  await expect(page.getByText("// resultados", { exact: true })).toBeVisible();
  await expect(page.getByRole("img").first()).toBeVisible();
  await shot(page, "vf-results.png");
  await page.locator("[data-tour='history']").click();
  await page.getByText("docs_code_tab", { exact: true }).first().click();
  await expect(page.getByRole("button", { name: "Baixar bundle do experimento" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Versionar com DVC" })).toBeVisible();
  await shot(page, "vf-run-detail.png");
}

const browser = await chromium.launch({
  ...(channel ? { channel } : {}),
  args: ["--disable-gpu"],
});
try {
  const context = await browser.newContext({
    baseURL,
    viewport: VIEWPORT,
    deviceScaleFactor: 1,
    locale: "pt-BR",
    colorScheme: "dark",
  });
  await context.addInitScript(() => {
    try {
      localStorage.setItem("vf.theme", "dark");
    } catch {
      // storage blocked: the dark colorScheme still applies
    }
  });
  const page = await context.newPage();
  page.setDefaultTimeout(30_000);
  await enter(page, "Researcher");
  await classification(page);
  await colabCorner(page);
  await replicates(page);
  await guideMenu(page);
  await codeTab(page);
  await resultsAndHistory(page);
  await context.close();
} finally {
  await browser.close();
}
