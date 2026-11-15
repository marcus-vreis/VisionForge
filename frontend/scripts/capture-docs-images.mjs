// Captures the README images from a running `visionforge gui` (issue #221).
//
// Driven by scripts/capture_docs_images.py, which starts the server in a
// throwaway work folder and home, and writes docs/images/README.md afterwards.
// Uses the Chrome already installed on the machine (channel "chrome"), as the
// local e2e suite does; set VF_DOCS_CHANNEL="" to use Playwright's chromium.
//
//   node scripts/capture-docs-images.mjs --base-url http://127.0.0.1:PORT --out DIR
//
// With --film DIR it also trains a second, longer run and saves the screen
// every few hundred milliseconds as numbered PNG frames in DIR; the Python
// script turns them into the animation of a training run.
import { chromium, expect } from "@playwright/test";
import { mkdirSync } from "node:fs";
import path from "node:path";
import { parseArgs } from "node:util";

const { values } = parseArgs({
  options: {
    "base-url": { type: "string" },
    out: { type: "string" },
    film: { type: "string" },
  },
});
const baseURL = values["base-url"];
const outDir = values.out;
const filmDir = values.film;
if (!baseURL || !outDir) {
  console.error("usage: capture-docs-images.mjs --base-url URL --out DIR");
  process.exit(2);
}
mkdirSync(outDir, { recursive: true });
if (filmDir) mkdirSync(filmDir, { recursive: true });

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

// The sample dataset's folder, read from the form once it is created.
let sampleDir = "";

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
  sampleDir = await page
    .locator("input")
    .evaluateAll((inputs) => inputs.find((i) => i.value.includes("exemplo-classificacao")).value);
}

async function classification(page) {
  await openTask(page, "Classificação");
  await useSampleDataset(page);
  await shot(page, "vf-classification.png");
  await openStep(page, "Estratégia");
  await expect(page.getByRole("button", { name: "Comparação replicada" })).toBeVisible();
  await shot(page, "vf-strategy.png");
  await modelAndImage(page);
}

/** Transfer learning switched on, then one preprocessing filter with its preview.
 *  Both are undone, so the run `codeTab` trains later is the default one. */
async function modelAndImage(page) {
  await openStep(page, "Modelo");
  const transfer = page.getByRole("switch", { name: "Transfer learning", exact: true });
  await transfer.click();
  await expect(page.getByText("// transfer learning", { exact: true })).toBeVisible();
  await shot(page, "vf-transfer.png");
  await transfer.click();
  await expect(transfer).toHaveText(/desativado/i);

  await openStep(page, "Imagem e aumento");
  await page.getByText("+ adicionar filtro", { exact: true }).click();
  await page.getByText("Unsharp mask", { exact: true }).click();
  await page.getByRole("button", { name: /Ver preview/ }).click();
  await expect(page.getByText("Original", { exact: true })).toBeVisible({ timeout: 30_000 });
  await shot(page, "vf-preprocessing.png");
  await page.getByRole("button", { name: "limpar", exact: true }).click();
  await expect(page.getByText(/Pipeline vazio/)).toBeVisible();
}

/** The dataset tools on the sample dataset. Leaves the overlay open: run it last. */
async function datasetExplorer(page) {
  // Esc leaves the run detail first, then the history itself.
  const history = page.getByTestId("history-overlay");
  await expect(async () => {
    await page.keyboard.press("Escape");
    await expect(history).toBeHidden({ timeout: 1_000 });
  }).toPass({ timeout: 15_000 });
  await page.locator("[data-tour='datasets']").click();
  await page.getByRole("tab", { name: "Explorar" }).click();
  // The same picker as the Dados step, which is hidden behind the window: the
  // visible one is the window's.
  await page.locator("input[placeholder*='datasets/coffee']:visible").fill(sampleDir);
  await page.getByRole("button", { name: "Abrir", exact: true }).click();
  // The window only chooses; the tools are in the full-screen lab.
  await page.getByRole("button", { name: "Abrir laboratório" }).click({ timeout: 30_000 });
  await expect(page.getByRole("heading", { name: "Visão geral" })).toBeVisible();
  await expect(page.getByText("Refazer divisão", { exact: true }).first()).toBeVisible();
  await shot(page, "vf-datasets-explorer.png");
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

/** The "Começando do zero" page: its opening, then the curves chapter. */
async function learnPage(page) {
  await page.getByTitle(/Abrir os guias/).click();
  await page.getByRole("menuitem", { name: /Começando do zero/ }).click();
  const learn = page.getByTestId("learn-overlay");
  await expect(learn.getByRole("heading", { level: 1 })).toBeVisible();
  await shot(page, "vf-learn.png");
  await learn.locator("#learn-curves").scrollIntoViewIfNeeded();
  await learn.locator("#learn-curves").evaluate((el) => el.scrollIntoView({ block: "start" }));
  await expect(learn.getByRole("button", { name: "adequada" })).toBeVisible();
  await shot(page, "vf-learn-curves.png");
  await page.keyboard.press("Escape");
  await expect(learn).toBeHidden();
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
  // The steps of the run beside the real code of the step shown (ADR-204).
  await sheet.getByRole("tab", { name: "Código ao vivo" }).click();
  const panel = sheet.locator("[aria-labelledby='training-tab-story']");
  await expect(panel).toBeVisible();
  await expect(panel.getByText("Época de treino", { exact: true }).first()).toBeVisible({ timeout: 30_000 });
  await expect(panel.getByText(/^def /).first()).toBeVisible({ timeout: 30_000 });
  await shot(page, "vf-code-tab.png");
  await sheet.getByRole("tab", { name: "Log" }).click();
  await shot(page, "vf-training.png");
}

/** The results of the run `codeTab` trained, then its entry in the history. */
async function resultsAndHistory(page) {
  const sheet = page.locator("[data-tour='training-sheet']");
  await sheet.getByRole("button", { name: /Ver resultados/ }).click();
  await expect(sheet).toBeHidden();
  // The finished run opens in History, which is its results page (ADR-201).
  const history = page.getByTestId("history-overlay");
  await expect(history).toBeVisible();
  await expect(history.getByText("Curvas por época", { exact: true })).toBeVisible();
  await shot(page, "vf-results.png");
  // The same detail, lower down: what can be done with the run.
  const bundle = history.getByRole("button", { name: "Baixar bundle do experimento" });
  await bundle.scrollIntoViewIfNeeded();
  await expect(bundle).toBeVisible();
  await expect(history.getByRole("button", { name: "Versionar com DVC" })).toBeVisible();
  await shot(page, "vf-run-detail.png");
}

// The film: epochs of the second run, the pause between frames and the most
// frames one run may take (a slow machine stops being filmed, not the run).
// The film is taken in a smaller window than the images, so the training
// sheet fills more of each frame; same 3:2 shape.
const FILM_VIEWPORT = { width: 1080, height: 720 };
const FILM_EPOCHS = "8";
const FILM_INTERVAL_MS = 250;
const FILM_MAX_FRAMES = 90;
let filmFrames = 0;

async function frame(page, times = 1) {
  await hideVersion(page);
  for (let i = 0; i < times; i += 1) {
    filmFrames += 1;
    const file = path.join(filmDir, `${String(filmFrames).padStart(3, "0")}.png`);
    await page.screenshot({ path: file });
  }
}

/** A second run of a few epochs, filmed from the Revisão step to its results. */
async function trainingFilm(page) {
  const history = page.getByTestId("history-overlay");
  await expect(async () => {
    await page.keyboard.press("Escape");
    await expect(history).toBeHidden({ timeout: 1_000 });
  }).toPass({ timeout: 15_000 });
  await openStep(page, "Treinamento");
  const epochs = field(page, "Épocas");
  await epochs.fill(FILM_EPOCHS);
  await epochs.press("Enter");
  await expect(field(page, "Épocas")).toHaveValue(FILM_EPOCHS);
  await openStep(page, "Revisão");
  await field(page, "Nome do experimento").fill("primeiro_treino");
  await page.setViewportSize(FILM_VIEWPORT);
  await page.mouse.move(0, 0);
  await page.waitForTimeout(1_200);
  await frame(page, 3);
  await page.locator("[data-tour='train']").click();
  const sheet = page.locator("[data-tour='training-sheet']");
  await expect(sheet).toBeVisible();
  const done = sheet.getByText("treino concluído", { exact: true });
  while (filmFrames < FILM_MAX_FRAMES && !(await done.isVisible())) {
    await frame(page);
    await page.waitForTimeout(FILM_INTERVAL_MS);
  }
  await expect(done).toBeVisible({ timeout: 300_000 });
  await page.waitForTimeout(600);
  await frame(page, 4);
  await sheet.getByRole("button", { name: /Ver resultados/ }).click();
  await expect(history).toBeVisible();
  await expect(history.getByText("Curvas por época", { exact: true })).toBeVisible();
  await page.waitForTimeout(1_200);
  await frame(page, 6);
  await page.setViewportSize(VIEWPORT);
  console.log(`filmed ${filmFrames} frames in ${filmDir}`);
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
  await learnPage(page);
  await codeTab(page);
  await resultsAndHistory(page);
  if (filmDir) await trainingFilm(page);
  await datasetExplorer(page);
  await context.close();
} finally {
  await browser.close();
}
